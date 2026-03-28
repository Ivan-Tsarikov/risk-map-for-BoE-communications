from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sentence_transformers import SentenceTransformer


LOOKUP_COLUMNS = [
    "story_key",
    "embedding_row",
    "text_for_embedding",
    "region",
    "week",
    "region_spread",
    "run_mode",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or reuse story embeddings.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--model-size", choices=["small", "base"], default=None)
    parser.add_argument("--run-mode", choices=["dev", "full"], default=None)
    parser.add_argument("--mode", choices=["build", "bootstrap-existing"], default="build")
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


def stable_hash(value: str, seed: int) -> int:
    payload = f"{seed}::{value}".encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16)


def allocate_proportional_quotas(bucket_counts: pd.DataFrame, cap: int) -> pd.DataFrame:
    quota_frame = bucket_counts.copy()
    quota_frame["quota"] = 0
    if quota_frame.empty or cap <= 0:
        return quota_frame

    total = int(quota_frame["story_count"].sum())
    if total <= cap:
        quota_frame["quota"] = quota_frame["story_count"].astype(int)
        return quota_frame

    raw_quota = quota_frame["story_count"] / total * cap
    quota_frame["quota"] = np.floor(raw_quota).astype(int)
    quota_frame["remainder"] = raw_quota - quota_frame["quota"]

    remaining = int(cap - quota_frame["quota"].sum())
    if remaining > 0:
        order = quota_frame.sort_values(
            ["remainder", "region", "week"],
            ascending=[False, True, True],
        ).index[:remaining]
        quota_frame.loc[order, "quota"] += 1

    return quota_frame.drop(columns=["remainder"])


def pick_story_sample(primary_buckets: pd.DataFrame, cap: int, strong_share: float, random_state: int) -> set[str]:
    if len(primary_buckets) <= cap:
        return set(primary_buckets["story_key"].astype(str))

    selected_keys: set[str] = set()
    strong_target = int(round(cap * strong_share))
    tier_targets = {
        "strong": strong_target,
        "weak_url": cap - strong_target,
    }
    tier_available = {
        tier: int(primary_buckets["sampling_tier"].eq(tier).sum()) for tier in tier_targets
    }

    for tier in ["strong", "weak_url"]:
        tier_targets[tier] = min(tier_targets[tier], tier_available[tier])

    leftover = cap - sum(tier_targets.values())
    for tier in ["strong", "weak_url"]:
        if leftover <= 0:
            break
        spare = tier_available[tier] - tier_targets[tier]
        if spare <= 0:
            continue
        add_now = min(spare, leftover)
        tier_targets[tier] += add_now
        leftover -= add_now

    for tier in ["strong", "weak_url"]:
        tier_frame = primary_buckets.loc[primary_buckets["sampling_tier"].eq(tier)].copy()
        if tier_frame.empty or tier_targets[tier] <= 0:
            continue

        bucket_counts = (
            tier_frame.groupby(["region", "week"], as_index=False)
            .agg(story_count=("story_key", "nunique"))
            .sort_values(["region", "week"])
        )
        quotas = allocate_proportional_quotas(bucket_counts, tier_targets[tier])
        tier_frame = tier_frame.merge(quotas, on=["region", "week"], how="left")
        tier_frame["selection_rank"] = tier_frame["story_key"].map(
            lambda value: stable_hash(f"{tier}::{value}", random_state)
        )
        tier_frame = tier_frame.sort_values(["region", "week", "selection_rank", "story_key"])
        tier_frame["rank_in_bucket"] = tier_frame.groupby(["region", "week"]).cumcount()
        tier_selected = tier_frame.loc[tier_frame["rank_in_bucket"] < tier_frame["quota"]]
        selected_keys.update(tier_selected["story_key"].tolist())

    return selected_keys


def load_story_inputs(candidate_path: Path, batch_size: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_frames = []
    meta_frames = []

    for chunk in iter_parquet_batches(
        candidate_path,
        batch_size=batch_size,
        columns=[
            "story_key",
            "region",
            "week",
            "unique_article_id",
            "text_for_embedding",
            "strong_keyword_hit",
        ],
    ):
        chunk["story_key"] = chunk["story_key"].astype(str)
        bucket_frames.append(
            chunk.groupby(["story_key", "region", "week"], as_index=False).agg(
                article_rows=("unique_article_id", "size")
            )
        )
        meta_frames.append(
            chunk.drop_duplicates("story_key")[
                ["story_key", "text_for_embedding", "strong_keyword_hit"]
            ]
        )

    if not bucket_frames:
        raise RuntimeError("candidate_articles.parquet is empty. Run 02_filter_candidates.py first.")

    story_buckets = pd.concat(bucket_frames, ignore_index=True)
    story_buckets = story_buckets.groupby(["story_key", "region", "week"], as_index=False).agg(
        article_rows=("article_rows", "sum")
    )

    story_meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates("story_key").copy()
    story_meta["sampling_tier"] = np.where(story_meta["strong_keyword_hit"], "strong", "weak_url")

    story_buckets["primary_rank"] = story_buckets.apply(
        lambda row: stable_hash(f"{row.story_key}::{row.region}::{row.week}", random_state),
        axis=1,
    )
    primary_buckets = story_buckets.sort_values(
        ["story_key", "article_rows", "primary_rank"],
        ascending=[True, False, True],
    ).drop_duplicates("story_key")

    region_spread = story_buckets.groupby("story_key", as_index=False).agg(
        region_spread=("region", "nunique")
    )
    story_meta = story_meta.merge(
        primary_buckets[["story_key", "region", "week"]],
        on="story_key",
        how="left",
    ).merge(region_spread, on="story_key", how="left")
    return story_meta, primary_buckets


def resolve_model_name(settings: dict, requested_size: str | None = None) -> tuple[str, str]:
    model_size = requested_size or settings["models"]["default_sentence_transformer_size"]
    model_lookup = {
        "small": settings["models"]["sentence_transformer_small"],
        "base": settings["models"]["sentence_transformer_base"],
    }
    if model_size not in model_lookup:
        raise ValueError(f"Unsupported model size: {model_size}")
    return model_lookup[model_size], model_size


def load_sentence_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(model: SentenceTransformer, texts: list[str], settings: dict) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return model.encode(
        texts,
        batch_size=settings["models"]["embedding_batch_size"],
        show_progress_bar=True,
        normalize_embeddings=settings["models"]["normalize_embeddings"],
        convert_to_numpy=True,
    ).astype("float32")


def encode_story_embeddings(
    model: SentenceTransformer,
    story_texts: list[str],
    settings: dict,
    output_path: Path,
) -> None:
    if output_path.exists():
        output_path.unlink()

    batch_size = settings["models"]["embedding_batch_size"] * 128
    embedding_store = None
    total = len(story_texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_vectors = encode_texts(model, story_texts[start:end], settings)
        if embedding_store is None:
            embedding_store = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype="float32",
                shape=(total, batch_vectors.shape[1]),
            )
        embedding_store[start:end] = batch_vectors
        print(f"  encoded_story_rows: {end}/{total}")

    if embedding_store is None:
        raise RuntimeError("No story embeddings were created.")

    del embedding_store


def compute_region_spread(candidate_path: Path, batch_size: int, selected_story_keys: set[str]) -> pd.DataFrame:
    spread_frames = []
    for chunk in iter_parquet_batches(candidate_path, batch_size=batch_size, columns=["story_key", "region"]):
        chunk["story_key"] = chunk["story_key"].astype(str)
        filtered = chunk.loc[chunk["story_key"].isin(selected_story_keys)]
        if not filtered.empty:
            spread_frames.append(
                filtered.groupby("story_key", as_index=False).agg(region_spread=("region", "nunique"))
            )

    if not spread_frames:
        return pd.DataFrame(columns=["story_key", "region_spread"])

    region_spread = pd.concat(spread_frames, ignore_index=True)
    return region_spread.groupby("story_key", as_index=False).agg(region_spread=("region_spread", "max"))


def bootstrap_existing_lookup(settings: dict) -> None:
    candidate_path = resolve_path(settings, "candidate_articles")
    story_catalog_path = resolve_path(settings, "story_catalog")
    story_embeddings_path = resolve_path(settings, "story_embeddings")
    story_lookup_path = ensure_parent(resolve_path(settings, "story_lookup"))

    if not story_catalog_path.exists():
        raise RuntimeError("Legacy story_catalog.parquet is missing, so bootstrap-existing cannot run.")
    if not story_embeddings_path.exists():
        raise RuntimeError("story_embeddings.npy is missing, so bootstrap-existing cannot run.")

    legacy_lookup = pd.read_parquet(
        story_catalog_path,
        columns=["story_key", "embedding_row", "text_for_embedding", "region", "week", "run_mode"],
    ).copy()
    legacy_lookup["story_key"] = legacy_lookup["story_key"].astype(str)
    selected_story_keys = set(legacy_lookup["story_key"])
    region_spread = compute_region_spread(
        candidate_path=candidate_path,
        batch_size=settings["runtime"]["chunk_size"],
        selected_story_keys=selected_story_keys,
    )
    story_lookup = legacy_lookup.merge(region_spread, on="story_key", how="left")
    story_lookup["region_spread"] = story_lookup["region_spread"].fillna(1).astype("int32")
    story_lookup = story_lookup[LOOKUP_COLUMNS].sort_values("embedding_row").reset_index(drop=True)

    embeddings = np.load(story_embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != len(story_lookup):
        raise RuntimeError(
            f"Bootstrap lookup has {len(story_lookup)} rows, but story_embeddings.npy has {embeddings.shape[0]} rows."
        )

    story_lookup.to_parquet(story_lookup_path, index=False)
    print("Embedding bootstrap summary")
    print(f"  source: {story_catalog_path}")
    print(f"  story_lookup_rows: {len(story_lookup)}")
    print(f"  story_embeddings_shape: {embeddings.shape}")
    print(f"  story_lookup: {story_lookup_path}")


def build_embeddings(settings: dict, args: argparse.Namespace) -> None:
    candidate_path = resolve_path(settings, "candidate_articles")
    story_lookup_path = ensure_parent(resolve_path(settings, "story_lookup"))
    story_embeddings_path = ensure_parent(resolve_path(settings, "story_embeddings"))

    if story_lookup_path.exists():
        story_lookup_path.unlink()
    if story_embeddings_path.exists():
        story_embeddings_path.unlink()

    batch_size = settings["runtime"]["chunk_size"]
    random_state = settings["runtime"]["random_state"]
    run_mode = args.run_mode or settings["runtime"]["run_mode"]
    model_name, model_size = resolve_model_name(settings, args.model_size)

    story_meta, primary_buckets = load_story_inputs(candidate_path, batch_size, random_state)
    total_story_count = len(story_meta)

    if run_mode == "full":
        selected_story_keys = set(story_meta["story_key"].astype(str))
    else:
        selected_story_keys = pick_story_sample(
            primary_buckets=primary_buckets.merge(
                story_meta[["story_key", "sampling_tier"]],
                on="story_key",
                how="left",
            ),
            cap=int(settings["runtime"]["embedding_story_cap"]),
            strong_share=float(settings["runtime"]["embedding_strong_share"]),
            random_state=random_state,
        )

    selected_stories = story_meta.loc[story_meta["story_key"].isin(selected_story_keys)].copy()
    selected_stories = selected_stories.sort_values("story_key").reset_index(drop=True)
    selected_stories["embedding_row"] = np.arange(len(selected_stories), dtype="int32")
    selected_stories["run_mode"] = run_mode

    print("Embedding stage summary")
    print(f"  run_mode: {run_mode}")
    print(f"  model_size: {model_size}")
    print(f"  model_name: {model_name}")
    print(f"  total_candidate_story_keys: {total_story_count}")
    print(f"  selected_story_keys: {len(selected_stories)}")
    if run_mode == "dev":
        print("  sampled_semantic_pilot: true")

    model = load_sentence_model(model_name)
    encode_story_embeddings(
        model=model,
        story_texts=selected_stories["text_for_embedding"].fillna("").astype(str).tolist(),
        settings=settings,
        output_path=story_embeddings_path,
    )

    story_lookup = selected_stories[LOOKUP_COLUMNS].copy()
    story_lookup["region_spread"] = story_lookup["region_spread"].fillna(1).astype("int32")
    story_lookup.to_parquet(story_lookup_path, index=False)
    print(f"  story_lookup: {story_lookup_path}")
    print(f"  story_embeddings: {story_embeddings_path}")


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    if args.mode == "bootstrap-existing":
        bootstrap_existing_lookup(settings)
        return

    build_embeddings(settings, args)


if __name__ == "__main__":
    main()
