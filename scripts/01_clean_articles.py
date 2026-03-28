from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


RAW_COLUMNS = [
    "unique_article_id",
    "article_text",
    "domain",
    "resolved_url",
    "article_author",
    "article_date",
    "article_id",
    "duplicate_group",
    "twitter",
    "twitter_id",
    "tweet_url",
    "tweet_date",
    "retweet_count",
    "reply_count",
    "like_count",
    "quote_count",
    "impression_count",
    "LAD",
    "main_LAD",
    "owner",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the regional news corpus.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for a smoke test.")
    return parser.parse_args()


def load_settings(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["project_root"] = config_path.resolve().parents[1]
    return settings


def resolve_path(settings: dict, key: str) -> Path:
    return Path(settings["project_root"]) / settings["paths"][key]


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def iter_csv_chunks(csv_path: Path, chunk_size: int, limit: int | None):
    rows_read = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if limit is not None:
            remaining = limit - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        rows_read += len(chunk)
        yield chunk
        if limit is not None and rows_read >= limit:
            break


def parse_multi_value_datetime(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = series.fillna("").astype(str).str.strip()
    has_multi_value = text.str.contains(";", regex=False, na=False)

    first_value = text.mask(text.eq(""), pd.NA)
    if has_multi_value.any():
        first_value.loc[has_multi_value] = first_value.loc[has_multi_value].apply(
            lambda value: min(part.strip() for part in value.split(";") if part.strip())
        )

    parsed = pd.to_datetime(first_value, errors="coerce", utc=True)
    return parsed, has_multi_value


def choose_primary_date(
    article_dates: pd.Series,
    tweet_dates: pd.Series,
    max_delta_days: int,
) -> tuple[pd.Series, pd.Series]:
    primary_date = tweet_dates.copy()
    date_source = pd.Series("tweet_date", index=primary_date.index, dtype=object)

    article_only = article_dates.notna() & tweet_dates.isna()
    primary_date.loc[article_only] = article_dates.loc[article_only]
    date_source.loc[article_only] = "article_date"

    both_dates = article_dates.notna() & tweet_dates.notna()
    if both_dates.any():
        delta_days = (
            (article_dates.loc[both_dates] - tweet_dates.loc[both_dates]).abs().dt.total_seconds() / 86400
        )
        use_article_date = delta_days <= max_delta_days
        safe_index = delta_days.index[use_article_date]
        primary_date.loc[safe_index] = article_dates.loc[safe_index]
        date_source.loc[safe_index] = "article_date"

    return primary_date, date_source


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def build_group_hash(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame[["story_key", "region", "week"]], index=False).astype("uint64")


def prepare_clean_chunk(chunk: pd.DataFrame, runtime: dict) -> tuple[pd.DataFrame, dict]:
    # Parse the two date sources and decide which one we trust for the final timeline.
    article_dates = pd.to_datetime(chunk["article_date"], errors="coerce", utc=True)
    tweet_dates, has_multi_tweet = parse_multi_value_datetime(chunk["tweet_date"])
    primary_dates, date_source = choose_primary_date(
        article_dates=article_dates,
        tweet_dates=tweet_dates,
        max_delta_days=runtime["date_max_delta_days"],
    )

    # Build the cleaned text and the final region used in all later aggregations.
    text_clean = clean_text(chunk["article_text"])
    text_length = text_clean.str.len()
    region = chunk["main_LAD"].fillna(chunk["LAD"]).fillna("").astype(str).str.strip()

    min_date = pd.Timestamp(runtime["cleaning_date_min"], tz="UTC")
    max_date = pd.Timestamp(runtime["cleaning_date_max"], tz="UTC")

    long_enough = text_length >= runtime["text_min_length"]
    has_region = region.ne("")
    in_date_range = primary_dates.between(min_date, max_date, inclusive="both")
    keep_row = primary_dates.notna() & in_date_range & long_enough & has_region

    clean_chunk = chunk.loc[keep_row].copy()
    if clean_chunk.empty:
        return clean_chunk, {
            "rows_seen": len(chunk),
            "rows_kept": 0,
            "rows_dropped_short_text": int((~long_enough).sum()),
            "rows_dropped_date_outlier": int((primary_dates.notna() & ~in_date_range).sum()),
            "rows_dropped_missing_date": int(primary_dates.isna().sum()),
            "rows_dropped_missing_region": int((~has_region).sum()),
            "multi_tweet_rows": int(has_multi_tweet.sum()),
            "missing_counts": {column: int(chunk[column].isna().sum()) for column in RAW_COLUMNS},
            "region_counts": {},
            "dup_counts": {},
            "examples": [],
            "clean_date_min": None,
            "clean_date_max": None,
        }

    # Keep both raw and cleaned columns so the notebook can show before/after examples.
    clean_chunk["date_raw_article"] = article_dates.loc[keep_row]
    clean_chunk["date_raw_tweet"] = tweet_dates.loc[keep_row]
    clean_chunk["date"] = primary_dates.loc[keep_row]
    clean_chunk["date_source"] = date_source.loc[keep_row]
    clean_chunk["region"] = region.loc[keep_row]
    clean_chunk["text_clean"] = text_clean.loc[keep_row]
    clean_chunk["text_len"] = text_length.loc[keep_row].astype("int32")
    clean_chunk["has_multi_tweet"] = has_multi_tweet.loc[keep_row].astype(bool)
    clean_chunk["story_key"] = (
        clean_chunk["duplicate_group"]
        .fillna(clean_chunk["article_id"])
        .fillna(clean_chunk["unique_article_id"])
        .astype(str)
        .str.strip()
    )
    clean_chunk["week"] = (
        clean_chunk["date"].dt.floor("D") - pd.to_timedelta(clean_chunk["date"].dt.weekday, unit="D")
    ).dt.tz_localize(None)
    clean_chunk["text_for_embedding"] = clean_chunk["text_clean"].str.slice(0, runtime["text_for_embedding_chars"])
    clean_chunk["group_hash"] = build_group_hash(clean_chunk)

    example_rows = []
    preview = clean_chunk[
        ["unique_article_id", "date", "region", "text_len", "article_text", "text_clean"]
    ].head(10)
    for row in preview.to_dict(orient="records"):
        example_rows.append(
            {
                "unique_article_id": row["unique_article_id"],
                "date": row["date"],
                "region": row["region"],
                "text_len": row["text_len"],
                "raw_preview": str(row["article_text"])[:220],
                "clean_preview": str(row["text_clean"])[:220],
            }
        )

    chunk_stats = {
        "rows_seen": len(chunk),
        "rows_kept": len(clean_chunk),
        "rows_dropped_short_text": int((~long_enough).sum()),
        "rows_dropped_date_outlier": int((primary_dates.notna() & ~in_date_range).sum()),
        "rows_dropped_missing_date": int(primary_dates.isna().sum()),
        "rows_dropped_missing_region": int((~has_region).sum()),
        "multi_tweet_rows": int(has_multi_tweet.sum()),
        "missing_counts": {column: int(chunk[column].isna().sum()) for column in RAW_COLUMNS},
        "region_counts": clean_chunk["region"].value_counts().to_dict(),
        "dup_counts": clean_chunk.groupby("group_hash", sort=False).size().to_dict(),
        "examples": example_rows,
        "clean_date_min": clean_chunk["date"].min(),
        "clean_date_max": clean_chunk["date"].max(),
    }
    return clean_chunk, chunk_stats


def update_global_stats(global_stats: dict, chunk_stats: dict) -> None:
    global_stats["rows_seen"] += chunk_stats["rows_seen"]
    global_stats["rows_kept"] += chunk_stats["rows_kept"]
    global_stats["rows_dropped_short_text"] += chunk_stats["rows_dropped_short_text"]
    global_stats["rows_dropped_date_outlier"] += chunk_stats["rows_dropped_date_outlier"]
    global_stats["rows_dropped_missing_date"] += chunk_stats["rows_dropped_missing_date"]
    global_stats["rows_dropped_missing_region"] += chunk_stats["rows_dropped_missing_region"]
    global_stats["multi_tweet_rows"] += chunk_stats["multi_tweet_rows"]

    for column, value in chunk_stats["missing_counts"].items():
        global_stats["missing_counts"][column] += value
    for region_name, value in chunk_stats["region_counts"].items():
        global_stats["region_counts"][region_name] += value
    for group_hash, value in chunk_stats["dup_counts"].items():
        global_stats["dup_counts"][group_hash] += value

    if chunk_stats["clean_date_min"] is not None:
        current_min = global_stats["clean_date_min"]
        current_max = global_stats["clean_date_max"]
        global_stats["clean_date_min"] = (
            chunk_stats["clean_date_min"] if current_min is None or chunk_stats["clean_date_min"] < current_min else current_min
        )
        global_stats["clean_date_max"] = (
            chunk_stats["clean_date_max"] if current_max is None or chunk_stats["clean_date_max"] > current_max else current_max
        )

    remaining_slots = 10 - len(global_stats["examples"])
    if remaining_slots > 0:
        global_stats["examples"].extend(chunk_stats["examples"][:remaining_slots])


def build_empty_stats() -> dict:
    return {
        "rows_seen": 0,
        "rows_kept": 0,
        "rows_dropped_short_text": 0,
        "rows_dropped_date_outlier": 0,
        "rows_dropped_missing_date": 0,
        "rows_dropped_missing_region": 0,
        "multi_tweet_rows": 0,
        "missing_counts": Counter(),
        "region_counts": Counter(),
        "dup_counts": Counter(),
        "examples": [],
        "clean_date_min": None,
        "clean_date_max": None,
    }


def write_cleaned_parquet(
    csv_path: Path,
    output_path: Path,
    runtime: dict,
    duplicate_counts: dict,
    limit: int | None,
) -> None:
    ensure_parent(output_path)
    if output_path.exists():
        output_path.unlink()

    writer = None

    try:
        for chunk in iter_csv_chunks(csv_path, runtime["chunk_size"], limit):
            clean_chunk, _ = prepare_clean_chunk(chunk, runtime)
            if clean_chunk.empty:
                continue

            # The hash is only an internal helper for counting duplicate stories across the whole corpus.
            clean_chunk["dup_weight"] = clean_chunk["group_hash"].map(duplicate_counts).rdiv(1.0).astype("float32")
            clean_chunk["date"] = clean_chunk["date"].dt.tz_localize(None)
            clean_chunk["date_raw_article"] = clean_chunk["date_raw_article"].dt.tz_localize(None)
            clean_chunk["date_raw_tweet"] = clean_chunk["date_raw_tweet"].dt.tz_localize(None)
            clean_chunk = clean_chunk.drop(columns=["group_hash"])

            table = pa.Table.from_pandas(clean_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def build_report(stats: dict, runtime: dict, limit: int | None) -> dict:
    total_rows = stats["rows_seen"]
    missing_share = {}
    for column in RAW_COLUMNS:
        if total_rows == 0:
            missing_share[column] = None
        else:
            missing_share[column] = round(stats["missing_counts"][column] / total_rows, 6)

    report = {
        "run_parameters": {
            "chunk_size": runtime["chunk_size"],
            "text_min_length": runtime["text_min_length"],
            "text_for_embedding_chars": runtime["text_for_embedding_chars"],
            "cleaning_date_min": runtime["cleaning_date_min"],
            "cleaning_date_max": runtime["cleaning_date_max"],
            "limit": limit,
        },
        "summary": {
            "rows_seen": stats["rows_seen"],
            "rows_kept": stats["rows_kept"],
            "rows_dropped_short_text": stats["rows_dropped_short_text"],
            "rows_dropped_date_outlier": stats["rows_dropped_date_outlier"],
            "rows_dropped_missing_date": stats["rows_dropped_missing_date"],
            "rows_dropped_missing_region": stats["rows_dropped_missing_region"],
            "multi_tweet_rows": stats["multi_tweet_rows"],
            "clean_date_min": stats["clean_date_min"],
            "clean_date_max": stats["clean_date_max"],
            "duplicate_story_groups": len(stats["dup_counts"]),
        },
        "missing_share": missing_share,
        "top_regions": stats["region_counts"].most_common(15),
        "examples": stats["examples"],
    }
    return json.loads(json.dumps(report, default=str))


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    runtime = settings["runtime"]

    csv_path = resolve_path(settings, "raw_articles")
    output_path = resolve_path(settings, "cleaned_articles")
    report_path = resolve_path(settings, "cleaning_report")

    global_stats = build_empty_stats()

    # First pass: profile the corpus and count duplicate stories by region-week.
    for chunk in iter_csv_chunks(csv_path, runtime["chunk_size"], args.limit):
        _, chunk_stats = prepare_clean_chunk(chunk, runtime)
        update_global_stats(global_stats, chunk_stats)

    # Second pass: write the cleaned parquet with the final duplicate weights.
    write_cleaned_parquet(
        csv_path=csv_path,
        output_path=output_path,
        runtime=runtime,
        duplicate_counts=dict(global_stats["dup_counts"]),
        limit=args.limit,
    )

    report = build_report(global_stats, runtime, args.limit)
    save_json(report_path, report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
