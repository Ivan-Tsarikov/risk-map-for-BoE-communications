from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REVIEW_COLUMNS = [
    "candidate_source",
    "unique_article_id",
    "date",
    "week",
    "region",
    "resolved_url",
    "strong_keyword_hit",
    "weak_keyword_hit",
    "url_section_hit",
    "text_clean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cheap high-recall candidate set for semantic scoring.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
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


def iter_parquet_batches(path: Path, batch_size: int, columns: list[str] | None = None):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def build_keyword_regex(terms: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(term) for term in terms]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def build_keyword_patterns(settings: dict) -> dict[str, re.Pattern[str]]:
    return {
        "strong": build_keyword_regex(settings["filters"]["strong_keywords"]),
        "weak": build_keyword_regex(settings["filters"]["weak_keywords"]),
        "url": build_keyword_regex(settings["filters"]["url_terms"]),
    }


def add_candidate_flags(frame: pd.DataFrame, patterns: dict[str, re.Pattern[str]]) -> pd.DataFrame:
    flagged = frame.copy()
    text = flagged["text_clean"].fillna("").astype(str)
    url = flagged["resolved_url"].fillna("").astype(str)

    flagged["strong_keyword_hit"] = text.str.contains(patterns["strong"], na=False)
    flagged["weak_keyword_hit"] = text.str.contains(patterns["weak"], na=False)
    flagged["url_section_hit"] = url.str.contains(patterns["url"], na=False)

    flagged["is_candidate"] = (
        flagged["strong_keyword_hit"] | flagged["weak_keyword_hit"] | flagged["url_section_hit"]
    )
    flagged["candidate_source"] = ""
    flagged.loc[flagged["strong_keyword_hit"], "candidate_source"] = "strong_keyword"
    flagged.loc[
        (~flagged["strong_keyword_hit"]) & flagged["weak_keyword_hit"] & flagged["url_section_hit"],
        "candidate_source",
    ] = "weak_keyword+url_section"
    flagged.loc[
        (~flagged["strong_keyword_hit"]) & flagged["weak_keyword_hit"] & (~flagged["url_section_hit"]),
        "candidate_source",
    ] = "weak_keyword"
    flagged.loc[
        (~flagged["strong_keyword_hit"]) & (~flagged["weak_keyword_hit"]) & flagged["url_section_hit"],
        "candidate_source",
    ] = "url_section"
    return flagged


def update_reservoir(sample_rows: list[dict], row: dict, max_size: int, seen: int, rng: random.Random) -> None:
    if len(sample_rows) < max_size:
        sample_rows.append(row)
        return

    replace_index = rng.randint(0, seen - 1)
    if replace_index < max_size:
        sample_rows[replace_index] = row


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    cleaned_path = resolve_path(settings, "cleaned_articles")
    candidate_path = ensure_parent(resolve_path(settings, "candidate_articles"))
    review_path = ensure_parent(resolve_path(settings, "candidate_review_sample"))

    if candidate_path.exists():
        candidate_path.unlink()

    batch_size = settings["runtime"]["chunk_size"]
    review_size = settings["runtime"]["review_sample_per_stratum"]
    rng = random.Random(settings["runtime"]["random_state"])
    patterns = build_keyword_patterns(settings)

    writer = None
    total_rows = 0
    candidate_rows = 0
    candidate_story_keys: set[str] = set()
    source_counts = {
        "strong_keyword": 0,
        "weak_keyword": 0,
        "url_section": 0,
        "weak_keyword+url_section": 0,
    }
    source_seen = {name: 0 for name in source_counts}
    source_samples = {name: [] for name in source_counts}

    try:
        for chunk in iter_parquet_batches(cleaned_path, batch_size=batch_size):
            total_rows += len(chunk)
            flagged = add_candidate_flags(chunk, patterns)
            candidate_chunk = flagged.loc[flagged["is_candidate"]].copy()
            if candidate_chunk.empty:
                continue

            candidate_rows += len(candidate_chunk)
            candidate_story_keys.update(candidate_chunk["story_key"].dropna().astype(str).unique().tolist())

            for source, frame in candidate_chunk.groupby("candidate_source"):
                source_counts[source] += len(frame)
                for row in frame[REVIEW_COLUMNS].to_dict(orient="records"):
                    source_seen[source] += 1
                    update_reservoir(
                        sample_rows=source_samples[source],
                        row=row,
                        max_size=review_size,
                        seen=source_seen[source],
                        rng=rng,
                    )

            table = pa.Table.from_pandas(candidate_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(candidate_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    review_rows: list[dict] = []
    for source in ["strong_keyword", "weak_keyword", "url_section", "weak_keyword+url_section"]:
        review_rows.extend(source_samples[source])
    pd.DataFrame(review_rows).to_csv(review_path, index=False, encoding="utf-8")

    print("Candidate filter summary")
    print(f"  total_cleaned_rows: {total_rows}")
    print(f"  candidate_rows: {candidate_rows}")
    print(f"  unique_candidate_story_keys: {len(candidate_story_keys)}")
    for source in ["strong_keyword", "weak_keyword", "url_section", "weak_keyword+url_section"]:
        print(f"  candidate_source__{source}: {source_counts[source]}")
    print(f"  candidate_review_sample: {review_path}")


if __name__ == "__main__":
    main()
