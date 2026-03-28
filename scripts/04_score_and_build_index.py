from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sentence_transformers import SentenceTransformer


REVIEW_COLUMNS = [
    "review_group",
    "unique_article_id",
    "date",
    "week",
    "region",
    "candidate_source",
    "assigned_topic",
    "top_topic",
    "top1_similarity",
    "econ_confidence",
    "resolved_url",
    "text_clean",
]

SCORED_COLUMNS = [
    "unique_article_id",
    "story_key",
    "date",
    "week",
    "region",
    "resolved_url",
    "text_clean",
    "dup_weight",
    "effective_weight",
    "is_econ",
    "econ_confidence",
    "assigned_topic",
    "top_topic",
    "top1_similarity",
    "topic_ambiguous",
    "region_spread",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score existing story embeddings and build the regional index.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--model-size", choices=["small", "base"], default=None)
    parser.add_argument("--reuse-embeddings", action="store_true")
    return parser.parse_args()


def load_settings(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["project_root"] = config_path.resolve().parents[1]
    return settings


def load_topic_profiles(settings: dict) -> list[dict]:
    topic_path = Path(settings["project_root"]) / "config" / "topic_profiles.yaml"
    with topic_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["topics"]


def resolve_path(settings: dict, key: str) -> Path:
    return Path(settings["project_root"]) / settings["paths"][key]


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def iter_parquet_batches(path: Path, batch_size: int, columns: list[str] | None = None):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def build_topic_text(topic: dict) -> str:
    keywords = ", ".join(topic["keywords"])
    examples = " ".join(topic["seed_examples"])
    return "\n".join(
        [
            topic["name"],
            topic["description"],
            f"Keywords: {keywords}",
            f"Seed examples: {examples}",
        ]
    )


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


def clipped_zscore(series: pd.Series, clip_value: float) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    zscore = (series - series.mean()) / std
    return zscore.clip(-clip_value, clip_value)


def rolling_previous_mean(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def calculate_persistence(group: pd.DataFrame) -> pd.Series:
    persistence = []
    run_length = 0
    for surprise in group["surprise"]:
        if surprise > 0:
            run_length += 1
        else:
            run_length = 0
        persistence.append(run_length)
    return pd.Series(persistence, index=group.index, dtype="int32")


def update_reservoir(sample_rows: list[dict], row: dict, max_size: int, seen: int, rng: random.Random) -> None:
    if len(sample_rows) < max_size:
        sample_rows.append(row)
        return

    replace_index = rng.randint(0, seen - 1)
    if replace_index < max_size:
        sample_rows[replace_index] = row


def load_story_lookup(settings: dict) -> tuple[pd.DataFrame, np.ndarray]:
    story_lookup_path = resolve_path(settings, "story_lookup")
    story_embeddings_path = resolve_path(settings, "story_embeddings")
    if not story_lookup_path.exists():
        raise RuntimeError("story_lookup.parquet is missing. Run 03_build_embeddings.py first.")
    if not story_embeddings_path.exists():
        raise RuntimeError("story_embeddings.npy is missing. Run 03_build_embeddings.py first.")

    story_lookup = pd.read_parquet(story_lookup_path).copy()
    story_lookup["story_key"] = story_lookup["story_key"].astype(str)
    story_lookup = story_lookup.sort_values("embedding_row").reset_index(drop=True)

    embeddings = np.load(story_embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != len(story_lookup):
        raise RuntimeError(
            f"story_lookup rows ({len(story_lookup)}) do not match story_embeddings rows ({embeddings.shape[0]})."
        )
    return story_lookup, embeddings


def load_story_flags(candidate_path: Path, batch_size: int, selected_story_keys: set[str]) -> pd.DataFrame:
    flag_frames = []
    for chunk in iter_parquet_batches(
        candidate_path,
        batch_size=batch_size,
        columns=["story_key", "candidate_source", "strong_keyword_hit"],
    ):
        chunk["story_key"] = chunk["story_key"].astype(str)
        filtered = chunk.loc[chunk["story_key"].isin(selected_story_keys)]
        if not filtered.empty:
            flag_frames.append(
                filtered.groupby("story_key", as_index=False).agg(
                    candidate_source=("candidate_source", "first"),
                    strong_keyword_hit=("strong_keyword_hit", "max"),
                )
            )

    if not flag_frames:
        raise RuntimeError("No candidate rows matched the selected story embeddings.")

    story_flags = pd.concat(flag_frames, ignore_index=True)
    return story_flags.groupby("story_key", as_index=False).agg(
        candidate_source=("candidate_source", "first"),
        strong_keyword_hit=("strong_keyword_hit", "max"),
    )


def score_story_embeddings(
    story_lookup: pd.DataFrame,
    embeddings: np.ndarray,
    settings: dict,
    topics: list[dict],
    model: SentenceTransformer,
    candidate_path: Path,
) -> pd.DataFrame:
    batch_size = settings["runtime"]["chunk_size"]
    selected_story_keys = set(story_lookup["story_key"])
    story_flags = load_story_flags(candidate_path, batch_size, selected_story_keys)

    story_scores = story_lookup.merge(story_flags, on="story_key", how="left")
    story_scores["strong_keyword_hit"] = story_scores["strong_keyword_hit"].fillna(False).astype(bool)

    positive_vectors = encode_texts(model, settings["profiles"]["econ_positive_anchors"], settings)
    negative_vectors = encode_texts(model, settings["profiles"]["econ_negative_anchors"], settings)
    positive_scores = embeddings @ positive_vectors.T
    negative_scores = embeddings @ negative_vectors.T
    econ_positive_max = positive_scores.max(axis=1)
    econ_negative_max = negative_scores.max(axis=1)
    econ_margin = econ_positive_max - econ_negative_max

    positive_rule = (
        (econ_positive_max >= settings["models"]["econ_min_similarity"])
        & (econ_margin >= settings["models"]["econ_margin_threshold"])
    )
    strong_rule = story_scores["strong_keyword_hit"].to_numpy() & (
        (econ_positive_max >= settings["models"]["econ_strong_min_similarity"])
        & (econ_margin >= settings["models"]["econ_strong_margin_threshold"])
    )
    negative_guard = econ_negative_max > (econ_positive_max + settings["models"]["econ_negative_guard"])

    is_econ = (positive_rule | strong_rule) & (~negative_guard)

    topic_names = [topic["name"] for topic in topics]
    topic_vectors = encode_texts(model, [build_topic_text(topic) for topic in topics], settings)
    similarity = embeddings @ topic_vectors.T
    top1_index = similarity.argmax(axis=1)
    top1_similarity = similarity[np.arange(len(similarity)), top1_index]
    sorted_similarity = np.sort(similarity, axis=1)
    top2_similarity = sorted_similarity[:, -2] if similarity.shape[1] > 1 else np.zeros(len(similarity))
    top_topic = np.array([topic_names[idx] for idx in top1_index], dtype=object)
    topic_ambiguous = (top1_similarity - top2_similarity) < settings["models"]["topic_ambiguous_margin"]

    assigned_topic = np.where(
        ~is_econ,
        "unassigned",
        np.where(
            top1_similarity >= settings["models"]["topic_min_similarity"],
            top_topic,
            "other_econ",
        ),
    )

    story_scores["econ_confidence"] = econ_positive_max.astype("float32")
    story_scores["is_econ"] = is_econ.astype(bool)
    story_scores["top_topic"] = top_topic
    story_scores["top1_similarity"] = top1_similarity.astype("float32")
    story_scores["topic_ambiguous"] = topic_ambiguous.astype(bool)
    story_scores["assigned_topic"] = assigned_topic
    return story_scores[
        [
            "story_key",
            "candidate_source",
            "strong_keyword_hit",
            "region_spread",
            "econ_confidence",
            "is_econ",
            "top_topic",
            "top1_similarity",
            "topic_ambiguous",
            "assigned_topic",
        ]
    ].copy()


def build_review_group(frame: pd.DataFrame) -> pd.Series:
    groups = np.where(
        frame["assigned_topic"].eq("unassigned"),
        "rejected_candidate",
        np.where(frame["assigned_topic"].eq("other_econ"), "other_econ", "boe_topic"),
    )
    return pd.Series(groups, index=frame.index)


def write_scored_articles(
    settings: dict,
    story_scores: pd.DataFrame,
    candidate_path: Path,
    scored_articles_path: Path,
    review_path: Path,
) -> pd.DataFrame:
    batch_size = settings["runtime"]["chunk_size"]
    selected_story_keys = set(story_scores["story_key"])
    score_lookup = story_scores.set_index("story_key")

    if scored_articles_path.exists():
        scored_articles_path.unlink()

    writer = None
    aggregation_frames = []
    rng = random.Random(settings["runtime"]["random_state"])
    review_seen = {"boe_topic": 0, "other_econ": 0, "rejected_candidate": 0}
    review_samples = {name: [] for name in review_seen}

    try:
        for chunk in iter_parquet_batches(
            candidate_path,
            batch_size=batch_size,
            columns=[
                "unique_article_id",
                "story_key",
                "date",
                "week",
                "region",
                "resolved_url",
                "text_clean",
                "dup_weight",
                "candidate_source",
            ],
        ):
            chunk["story_key"] = chunk["story_key"].astype(str)
            article_chunk = chunk.loc[chunk["story_key"].isin(selected_story_keys)].copy()
            if article_chunk.empty:
                continue

            article_chunk = article_chunk.join(
                score_lookup[
                    [
                        "region_spread",
                        "econ_confidence",
                        "is_econ",
                        "top_topic",
                        "top1_similarity",
                        "topic_ambiguous",
                        "assigned_topic",
                    ]
                ],
                on="story_key",
                how="left",
                rsuffix="_score",
            )
            article_chunk["region_spread"] = article_chunk["region_spread"].fillna(1).astype("int32")
            article_chunk["effective_weight"] = (
                article_chunk["dup_weight"].astype("float32")
                / np.sqrt(article_chunk["region_spread"].clip(lower=1).astype("float32"))
            )
            article_chunk["review_group"] = build_review_group(article_chunk)

            for review_group, frame in article_chunk.groupby("review_group"):
                for row in frame[REVIEW_COLUMNS].to_dict(orient="records"):
                    review_seen[review_group] += 1
                    update_reservoir(
                        sample_rows=review_samples[review_group],
                        row=row,
                        max_size=settings["runtime"]["review_sample_per_stratum"],
                        seen=review_seen[review_group],
                        rng=rng,
                    )

            scored_chunk = article_chunk[SCORED_COLUMNS].copy()
            table = pa.Table.from_pandas(scored_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(scored_articles_path, table.schema, compression="snappy")
            writer.write_table(table)

            aggregation_frames.append(
                scored_chunk[
                    [
                        "week",
                        "region",
                        "assigned_topic",
                        "is_econ",
                        "effective_weight",
                        "top1_similarity",
                    ]
                ].copy()
            )
    finally:
        if writer is not None:
            writer.close()

    review_rows: list[dict] = []
    for group in ["boe_topic", "other_econ", "rejected_candidate"]:
        review_rows.extend(review_samples[group])
    pd.DataFrame(review_rows).to_csv(review_path, index=False, encoding="utf-8")

    if not aggregation_frames:
        raise RuntimeError("No selected candidate rows were written to scored_articles.parquet.")

    return pd.concat(aggregation_frames, ignore_index=True)


def build_index(index_input: pd.DataFrame, topics: list[dict], settings: dict) -> pd.DataFrame:
    topic_names = [topic["name"] for topic in topics] + ["other_econ"]
    active_rows = index_input.loc[index_input["is_econ"]].copy()
    if active_rows.empty:
        raise RuntimeError("No economic article rows were found after scoring.")

    active_rows["week"] = pd.to_datetime(active_rows["week"])

    econ_totals = active_rows.groupby(["week", "region"], as_index=False).agg(
        total_econ_weight=("effective_weight", "sum")
    )

    topic_rows = active_rows.loc[active_rows["assigned_topic"].ne("unassigned")].copy()
    topic_rows["weighted_similarity"] = topic_rows["top1_similarity"] * topic_rows["effective_weight"]
    topic_agg = topic_rows.groupby(["week", "region", "assigned_topic"], as_index=False).agg(
        n_articles=("effective_weight", "sum"),
        weighted_similarity=("weighted_similarity", "sum"),
    )
    topic_agg["mean_similarity"] = np.where(
        topic_agg["n_articles"] > 0,
        topic_agg["weighted_similarity"] / topic_agg["n_articles"],
        0.0,
    )
    topic_agg = topic_agg.rename(columns={"assigned_topic": "topic"})
    topic_agg = topic_agg.drop(columns=["weighted_similarity"])

    active_region_weeks = econ_totals[["week", "region"]].drop_duplicates().copy()
    topic_grid = pd.DataFrame({"topic": topic_names})
    panel = active_region_weeks.assign(_key=1).merge(topic_grid.assign(_key=1), on="_key").drop(columns="_key")
    panel = panel.merge(topic_agg, on=["week", "region", "topic"], how="left")
    panel = panel.merge(econ_totals, on=["week", "region"], how="left")
    panel["n_articles"] = panel["n_articles"].fillna(0.0)
    panel["mean_similarity"] = panel["mean_similarity"].fillna(0.0)
    panel["total_econ_weight"] = panel["total_econ_weight"].fillna(0.0)
    panel["topic_share"] = np.where(
        panel["total_econ_weight"] > 0,
        panel["n_articles"] / panel["total_econ_weight"],
        0.0,
    )

    total_topic_weight = panel.groupby("topic", as_index=False).agg(topic_weight=("n_articles", "sum"))
    total_econ_weight = float(panel["total_econ_weight"].sum())
    topic_priors = total_topic_weight.copy()
    topic_priors["mu"] = np.where(
        total_econ_weight > 0,
        topic_priors["topic_weight"] / total_econ_weight,
        0.0,
    )
    prior_lookup = topic_priors.set_index("topic")["mu"].to_dict()
    prior_strength = float(settings["risk"]["prior_strength"])

    panel["topic_prior"] = panel["topic"].map(prior_lookup).fillna(0.0)
    panel["p_post"] = (
        panel["n_articles"] + panel["topic_prior"] * prior_strength
    ) / (panel["total_econ_weight"] + prior_strength)

    panel = panel.sort_values(["region", "topic", "week"]).reset_index(drop=True)
    panel["baseline_share"] = panel.groupby(["region", "topic"], sort=False)["p_post"].transform(
        lambda series: rolling_previous_mean(series, window=8, min_periods=3)
    )
    panel["baseline_share"] = panel["baseline_share"].fillna(panel["topic_prior"])
    panel["recent_share"] = panel.groupby(["region", "topic"], sort=False)["p_post"].transform(
        lambda series: rolling_previous_mean(series, window=4, min_periods=2)
    )
    panel["recent_share"] = panel["recent_share"].fillna(panel["baseline_share"])
    panel["surprise"] = panel["p_post"] - panel["baseline_share"]
    panel["momentum"] = panel["p_post"] - panel["recent_share"]
    panel["persistence_weeks"] = (
        panel.groupby(["region", "topic"], group_keys=False)
        .apply(calculate_persistence, include_groups=False)
        .astype("int32")
    )

    zscore_clip = settings["risk"]["zscore_clip"]
    weights = settings["risk"]["weights"]
    panel["surprise_z"] = panel.groupby("topic")["surprise"].transform(
        lambda series: clipped_zscore(series, zscore_clip)
    )
    panel["momentum_z"] = panel.groupby("topic")["momentum"].transform(
        lambda series: clipped_zscore(series, zscore_clip)
    )
    panel["mean_similarity_z"] = panel.groupby("topic")["mean_similarity"].transform(
        lambda series: clipped_zscore(series, zscore_clip)
    )
    support_k = float(settings["risk"]["support_k"])
    panel["support"] = np.sqrt(panel["total_econ_weight"] / (panel["total_econ_weight"] + support_k))
    panel["risk_score"] = panel["support"] * (
        weights["surprise_z"] * panel["surprise_z"]
        + weights["momentum_z"] * panel["momentum_z"]
        + weights["mean_similarity_z"] * panel["mean_similarity_z"]
    )

    return panel[
        [
            "week",
            "region",
            "topic",
            "n_articles",
            "total_econ_weight",
            "topic_share",
            "mean_similarity",
            "baseline_share",
            "surprise",
            "momentum",
            "persistence_weeks",
            "risk_score",
        ]
    ].copy()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    topics = load_topic_profiles(settings)
    model_name, model_size = resolve_model_name(settings, args.model_size)

    candidate_path = resolve_path(settings, "candidate_articles")
    scored_articles_path = ensure_parent(resolve_path(settings, "scored_articles"))
    index_path = ensure_parent(resolve_path(settings, "regional_topic_index"))
    review_path = ensure_parent(resolve_path(settings, "econ_review_sample"))

    if scored_articles_path.exists():
        scored_articles_path.unlink()
    if index_path.exists():
        index_path.unlink()

    story_lookup, embeddings = load_story_lookup(settings)
    print("Scoring stage summary")
    print(f"  reuse_embeddings: {bool(args.reuse_embeddings)}")
    print(f"  model_size: {model_size}")
    print(f"  model_name: {model_name}")
    print(f"  embedded_story_keys: {len(story_lookup)}")

    model = load_sentence_model(model_name)
    story_scores = score_story_embeddings(
        story_lookup=story_lookup,
        embeddings=embeddings,
        settings=settings,
        topics=topics,
        model=model,
        candidate_path=candidate_path,
    )
    print(f"  economic_story_keys: {int(story_scores['is_econ'].sum())}")
    print(f"  other_econ_story_keys: {int(story_scores['assigned_topic'].eq('other_econ').sum())}")
    print(f"  unassigned_story_keys: {int(story_scores['assigned_topic'].eq('unassigned').sum())}")

    index_input = write_scored_articles(
        settings=settings,
        story_scores=story_scores,
        candidate_path=candidate_path,
        scored_articles_path=scored_articles_path,
        review_path=review_path,
    )
    regional_index = build_index(index_input=index_input, topics=topics, settings=settings)
    regional_index.to_parquet(index_path, index=False)

    print(f"  scored_articles: {scored_articles_path}")
    print(f"  regional_topic_index: {index_path}")
    print(f"  econ_review_sample: {review_path}")


if __name__ == "__main__":
    main()
