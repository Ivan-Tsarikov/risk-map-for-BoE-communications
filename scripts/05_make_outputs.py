from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


BOILERPLATE_PATTERNS = [
    r"sign up to (our|the) [^.?!]{0,120}",
    r"subscribe [^.?!]{0,120}",
    r"read more[: ]?",
    r"image: [^.?!]{0,120}",
    r"share your email [^.?!]{0,120}",
    r"watch more of our videos [^.?!]{0,120}",
    r"visit shots! now",
]

UK_LAD_2023_WFS = (
    "https://dservices1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/services/"
    "Local_Authority_Districts_December_2023_Boundaries_UK_BUC/WFSServer"
)
UK_LAD_2023_TYPENAME = "Local_Authority_Districts_December_2023_Boundaries_UK_BUC:LAD_DEC_2023_UK_BUC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact MVP tables and figures.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    return parser.parse_args()


def load_settings(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    settings["project_root"] = config_path.resolve().parents[1]
    return settings


def load_boe_topic_names(settings: dict) -> list[str]:
    topic_path = Path(settings["project_root"]) / "config" / "topic_profiles.yaml"
    with topic_path.open("r", encoding="utf-8") as handle:
        topics = yaml.safe_load(handle)["topics"]
    return [topic["name"] for topic in topics]


def resolve_path(settings: dict, key: str) -> Path:
    return Path(settings["project_root"]) / settings["paths"][key]


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clean_snippet_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
    return cleaned


def extract_first_sentence(text: str, limit: int = 220) -> str:
    cleaned = clean_snippet_text(text)
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    first_sentence = sentences[0].strip()
    if 40 <= len(first_sentence) <= limit:
        return first_sentence
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def normalize_region_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").replace("’", "'").strip()).casefold()


def choose_reference_week(boe_index: pd.DataFrame) -> pd.Timestamp:
    active_weeks = boe_index.loc[boe_index["n_articles"] > 0, "week"].drop_duplicates().sort_values()
    for year in [2022, 2021]:
        year_weeks = active_weeks.loc[active_weeks.dt.year.eq(year)]
        if not year_weeks.empty:
            return year_weeks.max()
    if active_weeks.empty:
        raise RuntimeError("No active BoE weeks found in the regional index.")
    return active_weeks.max()


def choose_showcase_topic(reference_pairs: pd.DataFrame) -> str:
    topic_rank = (
        reference_pairs.groupby("topic", as_index=False)
        .agg(
            total_risk=("risk_score", lambda series: float(series.clip(lower=0).sum())),
            active_regions=("region", "nunique"),
            max_risk=("risk_score", "max"),
        )
        .sort_values(["total_risk", "active_regions", "max_risk", "topic"], ascending=[False, False, False, True])
    )
    if topic_rank.empty:
        raise RuntimeError("No active BoE topic found for the reference week.")
    return topic_rank.iloc[0]["topic"]


def choose_trend_regions(topic_window: pd.DataFrame, reference_week: pd.Timestamp, top_n: int = 5) -> list[str]:
    reference_scores = (
        topic_window.loc[topic_window["week"].eq(reference_week), ["region", "risk_score"]]
        .rename(columns={"risk_score": "reference_week_risk"})
    )
    ranked_regions = (
        topic_window.groupby("region", as_index=False)
        .agg(
            active_weeks=("n_articles", lambda series: int((series > 0).sum())),
            positive_risk=("risk_score", lambda series: float(series.clip(lower=0).sum())),
            avg_topic_share=("topic_share", "mean"),
        )
        .merge(reference_scores, on="region", how="left")
        .fillna({"reference_week_risk": 0.0})
        .sort_values(
            ["active_weeks", "positive_risk", "reference_week_risk", "avg_topic_share", "region"],
            ascending=[False, False, False, False, True],
        )
    )
    return ranked_regions.head(top_n)["region"].tolist()


def load_boundary_geojson(project_root: Path) -> dict:
    cache_path = project_root / "data" / "external" / "uk_lad_2023_buc.geojson"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    params = urllib.parse.urlencode(
        {
            "service": "wfs",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": UK_LAD_2023_TYPENAME,
            "outputFormat": "geojson",
        }
    )
    with urllib.request.urlopen(f"{UK_LAD_2023_WFS}?{params}", timeout=120) as response:
        geojson = json.load(response)

    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(geojson, handle)
    return geojson


def geometry_to_patches(geometry: dict) -> list[Polygon]:
    patches: list[Polygon] = []
    if geometry is None:
        return patches

    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return patches

    for polygon in polygons:
        if not polygon:
            continue
        outer_ring = polygon[0]
        patches.append(Polygon(outer_ring, closed=True))
    return patches


def draw_regional_map(
    boundary_geojson: dict,
    region_values: dict[str, float],
    topic_name: str,
    period_label: str,
    top_regions: pd.DataFrame,
    output_path: Path,
) -> None:
    patches: list[Polygon] = []
    patch_values: list[float] = []

    for feature in boundary_geojson["features"]:
        properties = feature.get("properties", {})
        region_name = properties.get("LAD23NM") or properties.get("LAD22NM")
        value = float(region_values.get(normalize_region_name(region_name), 0.0))
        for patch in geometry_to_patches(feature.get("geometry")):
            patches.append(patch)
            patch_values.append(value)

    if not patches:
        raise RuntimeError("Boundary file did not produce any map patches.")

    figure, axis = plt.subplots(figsize=(10, 16))
    cmap = plt.get_cmap("RdBu_r")
    if patch_values:
        value_min = min(patch_values)
        value_max = max(patch_values)
    else:
        value_min, value_max = -1.0, 1.0

    if value_min >= 0:
        value_min = min(-0.1, -0.05 * value_max if value_max else -0.1)
    if value_max <= 0:
        value_max = max(0.1, 0.05 * abs(value_min) if value_min else 0.1)

    norm = mcolors.TwoSlopeNorm(vmin=value_min, vcenter=0.0, vmax=value_max)

    collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.7,
    )
    collection.set_array(np.asarray(patch_values))
    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.axis("off")

    colorbar = figure.colorbar(collection, ax=axis, orientation="horizontal", fraction=0.03, pad=0.02)
    colorbar.ax.set_title("Risk Score: cold -> neutral -> hot", fontsize=11, loc="left", pad=14)

    axis.set_title(
        f"Regional Risk Map\n{topic_name} | {period_label}",
        fontsize=15,
        weight="bold",
        pad=18,
    )

    # Add a compact ranked legend so the map is readable without a separate table.
    ranked_lines = ["Top risk regions"]
    for rank, (_, row) in enumerate(top_regions.iterrows(), start=1):
        line = f"{rank}. {row['region']} ({row['risk_score']:.2f})"
        if "top_topic" in row.index and pd.notna(row["top_topic"]):
            line += f" - {row['top_topic']}"
        ranked_lines.append(line)
    axis.text(
        0.02,
        0.965,
        "\n".join(ranked_lines),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_topic_leader_map(
    boundary_geojson: dict,
    region_topics: dict[str, str],
    top_topic_table: pd.DataFrame,
    period_label: str,
    output_path: Path,
) -> None:
    topic_order = top_topic_table["top_topic"].dropna().drop_duplicates().tolist()
    palette = sns.color_palette("tab10", n_colors=max(len(topic_order), 3))
    color_lookup = {topic: palette[idx] for idx, topic in enumerate(topic_order)}
    color_lookup["No positive topic signal"] = (0.88, 0.88, 0.88)

    patches: list[Polygon] = []
    patch_colors: list[tuple[float, float, float]] = []

    for feature in boundary_geojson["features"]:
        properties = feature.get("properties", {})
        region_name = properties.get("LAD23NM") or properties.get("LAD22NM")
        topic = region_topics.get(normalize_region_name(region_name), "No positive topic signal")
        color = color_lookup.get(topic, color_lookup["No positive topic signal"])
        for patch in geometry_to_patches(feature.get("geometry")):
            patches.append(patch)
            patch_colors.append(color)

    if not patches:
        raise RuntimeError("Boundary file did not produce any map patches.")

    figure, axis = plt.subplots(figsize=(10, 16))
    collection = PatchCollection(
        patches,
        facecolor=patch_colors,
        edgecolor="white",
        linewidth=0.7,
    )
    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(
        f"Dominant Topic by Region\nreference week {period_label}",
        fontsize=15,
        weight="bold",
        pad=18,
    )

    legend_lines = ["Topic colors"]
    for _, row in top_topic_table.iterrows():
        legend_lines.append(f"{row['top_topic']}: {int(row['regions'])} regions")
    legend_lines.append("No positive topic signal: neutral/negative everywhere")
    axis.text(
        0.02,
        0.06,
        "\n".join(legend_lines),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12, markerfacecolor=color_lookup[topic], markeredgecolor=color_lookup[topic])
        for topic in topic_order + ["No positive topic signal"]
    ]
    labels = topic_order + ["No positive topic signal"]
    axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=10)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_trend_plot(
    topic_window: pd.DataFrame,
    selected_regions: list[str],
    topic_name: str,
    reference_week: pd.Timestamp,
    output_path: Path,
) -> None:
    plot_df = topic_window.loc[topic_window["region"].isin(selected_regions)].copy()
    benchmark = (
        topic_window.groupby("week", as_index=False)
        .agg(topic_share=("topic_share", "mean"), risk_score=("risk_score", "mean"))
        .sort_values("week")
    )
    color_map = dict(zip(selected_regions, sns.color_palette("Set2", n_colors=len(selected_regions))))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    sns.lineplot(
        data=plot_df,
        x="week",
        y="topic_share",
        hue="region",
        hue_order=selected_regions,
        palette=color_map,
        marker="o",
        linewidth=2.2,
        ax=axes[0],
    )
    axes[0].plot(
        benchmark["week"],
        benchmark["topic_share"],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="UK mean",
    )
    axes[0].axvline(reference_week, color="#666666", linestyle=":", linewidth=1.4)
    axes[0].set_ylabel("Topic share")
    axes[0].set_title(f"Trend of {topic_name} across the strongest regions", fontsize=14, weight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    sns.lineplot(
        data=plot_df,
        x="week",
        y="risk_score",
        hue="region",
        hue_order=selected_regions,
        palette=color_map,
        marker="o",
        linewidth=2.2,
        legend=False,
        ax=axes[1],
    )
    axes[1].plot(
        benchmark["week"],
        benchmark["risk_score"],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="UK mean",
    )
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1].axvline(reference_week, color="#666666", linestyle=":", linewidth=1.4)
    axes[1].set_ylabel("Risk score")
    axes[1].set_xlabel("Week")
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=3, loc="upper left", frameon=False)
    axes[1].legend(loc="upper left", frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def draw_overall_region_trend_plot(
    boe_index: pd.DataFrame,
    trend_weeks: list[pd.Timestamp],
    reference_week: pd.Timestamp,
    output_path: Path,
) -> None:
    window_df = boe_index.loc[boe_index["week"].isin(trend_weeks)].copy()
    window_df["positive_risk"] = window_df["risk_score"].clip(lower=0.0)

    region_totals = (
        window_df.groupby(["week", "region"], as_index=False)
        .agg(
            overall_risk=("positive_risk", "sum"),
            active_topics=("n_articles", lambda series: int((series > 0).sum())),
            mean_topic_share=("topic_share", "mean"),
        )
        .sort_values(["region", "week"])
    )

    top_regions = (
        region_totals.groupby("region", as_index=False)
        .agg(
            total_risk=("overall_risk", "sum"),
            active_weeks=("overall_risk", lambda series: int((series > 0).sum())),
            last_risk=("overall_risk", "last"),
        )
        .sort_values(["total_risk", "active_weeks", "last_risk", "region"], ascending=[False, False, False, True])
        .head(5)["region"]
        .tolist()
    )

    plot_df = region_totals.loc[region_totals["region"].isin(top_regions)].copy()
    benchmark = (
        region_totals.groupby("week", as_index=False)
        .agg(
            overall_risk=("overall_risk", "mean"),
            active_topics=("active_topics", "mean"),
        )
        .sort_values("week")
    )
    color_map = dict(zip(top_regions, sns.color_palette("Dark2", n_colors=len(top_regions))))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.95]},
    )

    sns.lineplot(
        data=plot_df,
        x="week",
        y="overall_risk",
        hue="region",
        hue_order=top_regions,
        palette=color_map,
        marker="o",
        linewidth=2.2,
        ax=axes[0],
    )
    axes[0].plot(
        benchmark["week"],
        benchmark["overall_risk"],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="UK mean",
    )
    axes[0].axvline(reference_week, color="#666666", linestyle=":", linewidth=1.4)
    axes[0].set_ylabel("Overall positive risk")
    axes[0].set_title("Overall regional risk across all BoE topics", fontsize=14, weight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    sns.lineplot(
        data=plot_df,
        x="week",
        y="active_topics",
        hue="region",
        hue_order=top_regions,
        palette=color_map,
        marker="o",
        linewidth=2.2,
        legend=False,
        ax=axes[1],
    )
    axes[1].plot(
        benchmark["week"],
        benchmark["active_topics"],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="UK mean",
    )
    axes[1].axvline(reference_week, color="#666666", linestyle=":", linewidth=1.4)
    axes[1].set_ylabel("Active BoE topics")
    axes[1].set_xlabel("Week")
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=3, loc="upper left", frameon=False)
    axes[1].legend(loc="upper left", frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    boe_topics = load_boe_topic_names(settings)

    scored_path = resolve_path(settings, "scored_articles")
    index_path = resolve_path(settings, "regional_topic_index")
    hot_pairs_path = ensure_parent(resolve_path(settings, "hot_regions_topics"))
    snippets_path = ensure_parent(resolve_path(settings, "representative_snippets"))
    heatmap_path = ensure_parent(resolve_path(settings, "latest_heatmap"))
    monthly_heatmap_path = ensure_parent(resolve_path(settings, "monthly_heatmap"))
    overall_risk_map_path = ensure_parent(resolve_path(settings, "overall_risk_map"))
    dominant_topic_map_path = ensure_parent(resolve_path(settings, "dominant_topic_map"))
    trend_path = ensure_parent(resolve_path(settings, "hot_topic_trends"))
    overall_region_trend_path = ensure_parent(resolve_path(settings, "overall_region_trends"))

    top_n = settings["runtime"]["hot_pairs_top_n"]

    index_df = pd.read_parquet(index_path)
    index_df["week"] = pd.to_datetime(index_df["week"])
    boe_index = index_df.loc[index_df["topic"].isin(boe_topics)].copy()

    reference_week = choose_reference_week(boe_index)
    active_recent_weeks = (
        boe_index.loc[(boe_index["week"] <= reference_week) & (boe_index["n_articles"] > 0), "week"]
        .drop_duplicates()
        .sort_values()
        .tail(4)
        .tolist()
    )

    reference_pairs = boe_index.loc[(boe_index["week"] == reference_week) & (boe_index["n_articles"] > 0)].copy()

    latest_pairs = (
        reference_pairs
        .nlargest(top_n, "risk_score")
        .copy()
    )
    latest_pairs["window"] = "reference_week"

    rolling_pairs = (
        boe_index.loc[(boe_index["week"].isin(active_recent_weeks)) & (boe_index["n_articles"] > 0)]
        .groupby(["region", "topic"], as_index=False)
        .agg(
            week=("week", "max"),
            n_articles=("n_articles", "sum"),
            total_econ_weight=("total_econ_weight", "mean"),
            topic_share=("topic_share", "mean"),
            mean_similarity=("mean_similarity", "mean"),
            baseline_share=("baseline_share", "mean"),
            surprise=("surprise", "mean"),
            momentum=("momentum", "mean"),
            persistence_weeks=("persistence_weeks", "max"),
            risk_score=("risk_score", "mean"),
        )
        .nlargest(top_n, "risk_score")
        .copy()
    )
    rolling_pairs["window"] = "recent_active_weeks"

    hot_pairs = pd.concat([latest_pairs, rolling_pairs], ignore_index=True)
    hot_pairs.to_csv(hot_pairs_path, index=False, encoding="utf-8")

    showcase_topic = choose_showcase_topic(reference_pairs)
    trend_weeks = (
        boe_index.loc[(boe_index["week"] <= reference_week) & (boe_index["n_articles"] > 0), "week"]
        .drop_duplicates()
        .sort_values()
        .tail(12)
        .tolist()
    )
    showcase_window = boe_index.loc[
        (boe_index["topic"] == showcase_topic) & (boe_index["week"].isin(trend_weeks))
    ].copy()
    trend_regions = choose_trend_regions(showcase_window, reference_week, top_n=5)

    boundary_geojson = load_boundary_geojson(Path(settings["project_root"]))
    showcase_reference = boe_index.loc[
        (boe_index["week"] == reference_week) & (boe_index["topic"] == showcase_topic),
        ["region", "risk_score", "n_articles"],
    ].copy()
    top_regions = (
        showcase_reference.loc[showcase_reference["risk_score"] > 0]
        .nlargest(8, "risk_score")
        .reset_index(drop=True)
    )
    if top_regions.empty:
        top_regions = showcase_reference.nlargest(8, "risk_score").reset_index(drop=True)

    draw_regional_map(
        boundary_geojson=boundary_geojson,
        region_values={
            normalize_region_name(row["region"]): float(row["risk_score"])
            for _, row in showcase_reference.iterrows()
        },
        topic_name=showcase_topic,
        period_label=f"week of {reference_week.date().isoformat()}",
        top_regions=top_regions,
        output_path=heatmap_path,
    )

    december_window = boe_index.loc[
        (boe_index["topic"] == showcase_topic)
        & (boe_index["week"].dt.year.eq(2022))
        & (boe_index["week"].dt.month.eq(12))
    ].copy()
    december_map = (
        december_window.groupby("region", as_index=False)
        .agg(
            risk_score=("risk_score", "mean"),
            n_articles=("n_articles", "sum"),
        )
        .sort_values(["risk_score", "n_articles", "region"], ascending=[False, False, True])
    )
    december_top_regions = (
        december_map.loc[december_map["risk_score"] > 0]
        .head(8)
        .reset_index(drop=True)
    )
    if december_top_regions.empty:
        december_top_regions = december_map.head(8).reset_index(drop=True)

    draw_regional_map(
        boundary_geojson=boundary_geojson,
        region_values={
            normalize_region_name(row["region"]): float(row["risk_score"])
            for _, row in december_map.iterrows()
        },
        topic_name=showcase_topic,
        period_label="December 2022 average",
        top_regions=december_top_regions,
        output_path=monthly_heatmap_path,
    )

    region_topic_leaders = (
        reference_pairs.sort_values(["region", "risk_score", "topic"], ascending=[True, False, True])
        .groupby("region", as_index=False)
        .head(1)
        .rename(columns={"topic": "top_topic", "risk_score": "top_risk_score"})
        .reset_index(drop=True)
    )
    positive_leaders = region_topic_leaders.loc[region_topic_leaders["top_risk_score"] > 0].copy()
    leader_legend = (
        positive_leaders.groupby("top_topic", as_index=False)
        .agg(regions=("region", "nunique"), max_risk=("top_risk_score", "max"))
        .sort_values(["regions", "max_risk", "top_topic"], ascending=[False, False, True])
    )

    draw_regional_map(
        boundary_geojson=boundary_geojson,
        region_values={
            normalize_region_name(row["region"]): float(row["top_risk_score"])
            for _, row in region_topic_leaders.iterrows()
        },
        topic_name="Top BoE topic risk by region",
        period_label=f"reference week {reference_week.date().isoformat()}",
        top_regions=(
            region_topic_leaders.nlargest(8, "top_risk_score")
            .rename(columns={"top_risk_score": "risk_score"})
            [["region", "risk_score", "top_topic"]]
            .reset_index(drop=True)
        ),
        output_path=overall_risk_map_path,
    )
    draw_topic_leader_map(
        boundary_geojson=boundary_geojson,
        region_topics={
            normalize_region_name(row["region"]): (
                row["top_topic"] if row["top_risk_score"] > 0 else "No positive topic signal"
            )
            for _, row in region_topic_leaders.iterrows()
        },
        top_topic_table=leader_legend,
        period_label=reference_week.date().isoformat(),
        output_path=dominant_topic_map_path,
    )

    draw_trend_plot(
        topic_window=showcase_window,
        selected_regions=trend_regions,
        topic_name=showcase_topic,
        reference_week=reference_week,
        output_path=trend_path,
    )
    draw_overall_region_trend_plot(
        boe_index=boe_index,
        trend_weeks=trend_weeks,
        reference_week=reference_week,
        output_path=overall_region_trend_path,
    )

    scored_articles = pd.read_parquet(
        scored_path,
        columns=[
            "unique_article_id",
            "week",
            "region",
            "assigned_topic",
            "resolved_url",
            "text_clean",
            "top1_similarity",
            "region_spread",
        ],
    )
    scored_articles["week"] = pd.to_datetime(scored_articles["week"])
    snippet_pairs = latest_pairs[["week", "region", "topic", "risk_score", "n_articles", "topic_share"]].copy()
    snippet_candidates = scored_articles.merge(
        snippet_pairs,
        left_on=["week", "region", "assigned_topic"],
        right_on=["week", "region", "topic"],
        how="inner",
    )
    snippet_candidates["representative_snippet"] = snippet_candidates["text_clean"].map(
        lambda value: extract_first_sentence(value, limit=220)
    )
    snippet_candidates = snippet_candidates.loc[snippet_candidates["representative_snippet"].ne("")]
    snippet_candidates = snippet_candidates.sort_values(
        ["week", "region", "topic", "top1_similarity", "region_spread", "unique_article_id"],
        ascending=[True, True, True, False, True, True],
    )
    snippet_candidates["snippet_rank"] = (
        snippet_candidates.groupby(["week", "region", "topic"]).cumcount() + 1
    )
    snippet_candidates = snippet_candidates.loc[snippet_candidates["snippet_rank"] <= 3]

    snippet_candidates[
        [
            "week",
            "region",
            "topic",
            "risk_score",
            "n_articles",
            "topic_share",
            "snippet_rank",
            "unique_article_id",
            "resolved_url",
            "top1_similarity",
            "region_spread",
            "representative_snippet",
        ]
    ].to_csv(snippets_path, index=False, encoding="utf-8")

    print(f"reference_week: {reference_week.date()}")
    print(f"recent_active_weeks: {[week.date().isoformat() for week in active_recent_weeks]}")
    print(f"showcase_topic: {showcase_topic}")
    print(f"trend_regions: {trend_regions}")
    print(f"monthly_heatmap: {monthly_heatmap_path}")
    print(f"overall_risk_map: {overall_risk_map_path}")
    print(f"dominant_topic_map: {dominant_topic_map_path}")
    print(f"overall_region_trends: {overall_region_trend_path}")
    print(f"hot_pairs: {hot_pairs_path}")
    print(f"representative_snippets: {snippets_path}")


if __name__ == "__main__":
    main()
