"""Generate paper-facing Results values and figures for the walking corpus.

This script is intentionally independent of the runtime pipeline configuration.
It reads the authoritative walking pipeline state directly and derives all
paper-facing counts from completed records and their accepted segments.

Default repository layout
-------------------------
The current walking pipeline writes its state to::

    data/walking_pipeline_state.json

Run from the repository root::

    python analysis.py

Use a different state file::

    python analysis.py --state /path/to/walking_pipeline_state.json

Outputs
-------
The script writes the following under ``_output/chi_walking`` by default:

* ``paper_values.txt``: manuscript-ready counts and percentages.
* ``paper_values.json``: the same summary in machine-readable form.
* ``tables/*.csv``: detailed tables used to produce the reported values.
* ``figures/*.html``: interactive versions of every figure.
* ``figures/*.png`` and ``figures/*.pdf`` when Kaleido is available.

Publication copies of the generated figures are also written to
``figures/chi_walking`` by default.

Important denominator rules
---------------------------
* Corpus, environment, time-of-day, provenance, and duration statistics use
  accepted segments from videos whose final status is ``complete``.
* Video status statistics use every record in the state file.
* Geographic segment statistics inherit the video-level Nominatim resolution
  stored in the corresponding complete video record.
* Segment-specific geographic evidence is kept distinct from video-level
  title/description evidence. The script reports both so the manuscript can
  use precise wording.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent

TIME_OF_DAY_CODES = {
    0: "day",
    1: "night",
    2: "dawn_dusk",
    -1: "unknown",
}

TIME_OF_DAY_ORDER = ["day", "night", "dawn_dusk", "unknown"]
WALKING_ENVIRONMENT_ORDER = [
    "street",
    "indoor",
    "beach",
    "park_nature",
    "trail",
    "market",
    "waterfront",
    "square_plaza",
    "transport_hub",
    "mixed",
    "other",
    "unknown",
    "not_applicable",
]
LOCATION_SOURCE_ORDER = [
    "timestamp_description",
    "embedded_video",
    "both",
    "none",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse walking_pipeline_state.json and generate paper-facing "
            "Results values, tables, and figures."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "data" / "walking_pipeline_state.json",
        help=(
            "Path to walking_pipeline_state.json. "
            "Default: ./data/walking_pipeline_state.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_output" / "chi_walking",
        help="Directory for paper values, tables, and generated figures.",
    )
    parser.add_argument(
        "--publication-dir",
        type=Path,
        default=ROOT / "figures" / "chi_walking",
        help="Directory receiving copies of generated publication figures.",
    )
    parser.add_argument(
        "--top-countries",
        type=int,
        default=15,
        help="Number of countries shown in the country bar chart. Default: 15.",
    )
    parser.add_argument(
        "--map-top-environments",
        type=int,
        default=4,
        help=(
            "Number of walking-environment categories shown in geographic "
            "small multiples. Default: 4."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Compute paper values and tables without generating figures.",
    )
    return parser


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "none", "null", "nan"}:
        return ""
    return text


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentage(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _human_label(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return "Unknown"
    replacements = {
        "dawn_dusk": "Dawn or dusk",
        "park_nature": "Park or nature",
        "square_plaza": "Square or plaza",
        "transport_hub": "Transport hub",
        "timestamp_description": "Timestamp / chapter",
        "embedded_video": "Embedded video text",
        "not_applicable": "Not applicable",
    }
    if text in replacements:
        return replacements[text]
    return text.replace("_", " ").title()


def _normalise_time_of_day(value: Any) -> str:
    if isinstance(value, bool):
        return "unknown"

    if isinstance(value, int):
        return TIME_OF_DAY_CODES.get(value, "unknown")

    if isinstance(value, float) and value.is_integer():
        return TIME_OF_DAY_CODES.get(int(value), "unknown")

    text = _clean_text(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "day": "day",
        "daytime": "day",
        "night": "night",
        "nighttime": "night",
        "dawn": "dawn_dusk",
        "dusk": "dawn_dusk",
        "twilight": "dawn_dusk",
        "dawn_dusk": "dawn_dusk",
        "dawn_or_dusk": "dawn_dusk",
        "unknown": "unknown",
    }
    if text in aliases:
        return aliases[text]

    try:
        numeric = int(text)
    except (TypeError, ValueError):
        return "unknown"
    return TIME_OF_DAY_CODES.get(numeric, "unknown")


def _normalise_environment(value: Any) -> str:
    text = _clean_text(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "park_or_nature": "park_nature",
        "park/nature": "park_nature",
        "square_or_plaza": "square_plaza",
        "square/plaza": "square_plaza",
        "transporthub": "transport_hub",
    }
    text = aliases.get(text, text)
    return text if text else "unknown"


def _normalise_location_source(value: Any) -> str:
    text = _clean_text(value).casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "timestamp": "timestamp_description",
        "chapter": "timestamp_description",
        "timestamp_or_chapter": "timestamp_description",
        "description": "timestamp_description",
        "embedded": "embedded_video",
        "video": "embedded_video",
        "embedded_text": "embedded_video",
    }
    text = aliases.get(text, text)
    if text not in set(LOCATION_SOURCE_ORDER):
        return "none"
    return text


def _segment_duration_seconds(segment: dict[str, Any]) -> float:
    duration = _finite_float(segment.get("duration_seconds"))
    if duration is not None and duration >= 0:
        return duration

    start = _finite_float(segment.get("start_time"))
    end = _finite_float(segment.get("end_time"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def _text_decision_has_location_evidence(decision: Any) -> bool:
    if not isinstance(decision, dict):
        return False
    return any(
        _clean_text(decision.get(field))
        for field in ("locality", "state", "country")
    )


def _location_status(location: Any) -> str:
    if not isinstance(location, dict):
        return "missing_location_record"
    status = _clean_text(location.get("geocode_status"))
    if not status:
        return "missing_location_status"
    if status.casefold().startswith("failed:"):
        return "failed"
    return status.casefold()


def _is_publishable_location(location: Any) -> bool:
    if not isinstance(location, dict):
        return False
    if _location_status(location) != "resolved":
        return False

    locality = _clean_text(location.get("locality"))
    country = _clean_text(location.get("country"))
    lat = _finite_float(location.get("lat"))
    lon = _finite_float(location.get("lon"))

    return bool(
        locality
        and country
        and lat is not None
        and lon is not None
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    )


def _canonical_locality_key(location: dict[str, Any]) -> str:
    locality = _clean_text(location.get("locality")).casefold()
    state = _clean_text(location.get("state")).casefold()
    iso3 = _clean_text(location.get("iso3")).upper()
    country = _clean_text(location.get("country")).casefold()
    lat = _finite_float(location.get("lat"))
    lon = _finite_float(location.get("lon"))
    return "|".join(
        [
            locality,
            state,
            iso3 or country,
            "" if lat is None else f"{lat:.7f}",
            "" if lon is None else f"{lon:.7f}",
        ]
    )


def _ordered_count_table(
    values: Iterable[str],
    *,
    category_name: str,
    preferred_order: list[str] | None = None,
) -> pd.DataFrame:
    series = pd.Series(list(values), dtype="object")
    counter = Counter(series.tolist())
    total = sum(counter.values())

    order: list[str] = []
    if preferred_order:
        order.extend(value for value in preferred_order if value in counter)
    order.extend(
        key
        for key, _ in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if key not in order
    )

    rows = [
        {
            category_name: key,
            "label": _human_label(key),
            "count": int(counter[key]),
            "percentage": _percentage(counter[key], total),
        }
        for key in order
    ]
    return pd.DataFrame(
        rows,
        columns=[category_name, "label", "count", "percentage"],
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Walking state file not found: {path}\n"
            "Use --state to point to walking_pipeline_state.json."
        )
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or not isinstance(state.get("videos", {}), dict):
        raise ValueError("The state file does not contain a valid 'videos' object.")
    return state


def extract_segments(state: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for video_id, record in state.get("videos", {}).items():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue

        segments = record.get("segments", [])
        if not isinstance(segments, list):
            continue

        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        decision = (
            record.get("text_decision")
            if isinstance(record.get("text_decision"), dict)
            else {}
        )
        location = record.get("location") if isinstance(record.get("location"), dict) else {}

        video_metadata_evidence = _text_decision_has_location_evidence(decision)
        resolved = _is_publishable_location(location)
        locality_key = _canonical_locality_key(location) if resolved else ""

        for position, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue

            source = _normalise_location_source(segment.get("location_source"))
            timestamp_labels = segment.get("timestamp_labels")
            embedded_location_text = segment.get("embedded_location_text")

            has_timestamp = bool(timestamp_labels) or source in {
                "timestamp_description",
                "both",
            }
            has_embedded = bool(embedded_location_text) or source in {
                "embedded_video",
                "both",
            }
            segment_specific_evidence = has_timestamp or has_embedded

            rows.append(
                {
                    "video_id": str(video_id),
                    "segment_index": segment.get("segment_index", position),
                    "start_time": _finite_float(segment.get("start_time")),
                    "end_time": _finite_float(segment.get("end_time")),
                    "duration_seconds": _segment_duration_seconds(segment),
                    "time_of_day": _normalise_time_of_day(segment.get("time_of_day")),
                    "walking_environment": _normalise_environment(
                        segment.get("walking_environment")
                    ),
                    "location_source": source,
                    "has_timestamp_evidence": bool(has_timestamp),
                    "has_embedded_evidence": bool(has_embedded),
                    "has_segment_location_evidence": bool(segment_specific_evidence),
                    "has_video_metadata_location_evidence": bool(video_metadata_evidence),
                    "has_any_location_evidence": bool(
                        segment_specific_evidence or video_metadata_evidence
                    ),
                    "confidence": _finite_float(segment.get("confidence")),
                    "walking_fraction": _finite_float(segment.get("walking_fraction")),
                    "promotion_fraction": _finite_float(segment.get("promotion_fraction")),
                    "drone_aerial_fraction": _finite_float(segment.get("drone_aerial_fraction")),
                    "decision_method": _clean_text(segment.get("decision_method")),
                    "geocode_status": _location_status(location),
                    "resolved_location": bool(resolved),
                    "locality_key": locality_key,
                    "locality": _clean_text(location.get("locality")) if resolved else "",
                    "state": _clean_text(location.get("state")) if resolved else "",
                    "country": _clean_text(location.get("country")) if resolved else "",
                    "iso3": _clean_text(location.get("iso3")).upper() if resolved else "",
                    "continent": _clean_text(location.get("continent")) if resolved else "",
                    "lat": _finite_float(location.get("lat")) if resolved else None,
                    "lon": _finite_float(location.get("lon")) if resolved else None,
                    "upload_date": _clean_text(metadata.get("upload_date")),
                    "channel": _clean_text(metadata.get("channel")),
                }
            )

    columns = [
        "video_id",
        "segment_index",
        "start_time",
        "end_time",
        "duration_seconds",
        "time_of_day",
        "walking_environment",
        "location_source",
        "has_timestamp_evidence",
        "has_embedded_evidence",
        "has_segment_location_evidence",
        "has_video_metadata_location_evidence",
        "has_any_location_evidence",
        "confidence",
        "walking_fraction",
        "promotion_fraction",
        "drone_aerial_fraction",
        "decision_method",
        "geocode_status",
        "resolved_location",
        "locality_key",
        "locality",
        "state",
        "country",
        "iso3",
        "continent",
        "lat",
        "lon",
        "upload_date",
        "channel",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_video_status_table(state: dict[str, Any]) -> pd.DataFrame:
    statuses: list[str] = []
    for record in state.get("videos", {}).values():
        if not isinstance(record, dict):
            statuses.append("invalid_record")
        else:
            statuses.append(_clean_text(record.get("status")) or "missing")
    return _ordered_count_table(statuses, category_name="status")


def build_geocode_status_table(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(columns=["geocode_status", "label", "count", "percentage"])
    return _ordered_count_table(
        segments["geocode_status"].astype(str).tolist(),
        category_name="geocode_status",
    )


def build_environment_table(segments: pd.DataFrame) -> pd.DataFrame:
    return _ordered_count_table(
        segments["walking_environment"].astype(str).tolist(),
        category_name="walking_environment",
        preferred_order=WALKING_ENVIRONMENT_ORDER,
    )


def build_time_table(segments: pd.DataFrame) -> pd.DataFrame:
    return _ordered_count_table(
        segments["time_of_day"].astype(str).tolist(),
        category_name="time_of_day",
        preferred_order=TIME_OF_DAY_ORDER,
    )


def build_location_source_table(segments: pd.DataFrame) -> pd.DataFrame:
    return _ordered_count_table(
        segments["location_source"].astype(str).tolist(),
        category_name="location_source",
        preferred_order=LOCATION_SOURCE_ORDER,
    )


def build_location_evidence_table(segments: pd.DataFrame) -> pd.DataFrame:
    total = len(segments)
    timestamp_any = int(segments["has_timestamp_evidence"].sum()) if total else 0
    embedded_any = int(segments["has_embedded_evidence"].sum()) if total else 0
    both = int(
        (segments["has_timestamp_evidence"] & segments["has_embedded_evidence"]).sum()
    ) if total else 0
    timestamp_only = timestamp_any - both
    embedded_only = embedded_any - both
    neither = total - timestamp_only - embedded_only - both
    segment_specific = int(segments["has_segment_location_evidence"].sum()) if total else 0
    metadata = int(segments["has_video_metadata_location_evidence"].sum()) if total else 0
    any_evidence = int(segments["has_any_location_evidence"].sum()) if total else 0

    rows = [
        ("timestamp_only", timestamp_only),
        ("embedded_only", embedded_only),
        ("both", both),
        ("neither_segment_specific", neither),
        ("timestamp_any", timestamp_any),
        ("embedded_any", embedded_any),
        ("segment_specific_any", segment_specific),
        ("video_metadata_any", metadata),
        ("any_geographic_evidence", any_evidence),
    ]
    return pd.DataFrame(
        [
            {
                "evidence_measure": key,
                "label": _human_label(key),
                "count": int(count),
                "percentage_of_accepted_segments": _percentage(count, total),
            }
            for key, count in rows
        ]
    )


def build_country_table(resolved: pd.DataFrame) -> pd.DataFrame:
    if resolved.empty:
        return pd.DataFrame(columns=["country_key", "country", "segments", "percentage"])

    data = resolved.copy()
    data["country_key"] = data["iso3"].where(data["iso3"].astype(bool), data["country"])
    grouped = (
        data.groupby(["country_key", "country"], dropna=False)
        .size()
        .reset_index(name="segments")
        .sort_values(["segments", "country"], ascending=[False, True], ignore_index=True)
    )
    grouped["percentage"] = (
        grouped["segments"] / len(resolved) * 100.0
    ).round(2)
    return grouped


def build_continent_table(resolved: pd.DataFrame) -> pd.DataFrame:
    if resolved.empty:
        return pd.DataFrame(columns=["continent", "segments", "percentage"])
    grouped = (
        resolved.assign(continent=resolved["continent"].replace("", "Unknown"))
        .groupby("continent", dropna=False)
        .size()
        .reset_index(name="segments")
        .sort_values(["segments", "continent"], ascending=[False, True], ignore_index=True)
    )
    grouped["percentage"] = (
        grouped["segments"] / len(resolved) * 100.0
    ).round(2)
    return grouped


def build_locality_table(resolved: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "locality_key",
        "locality",
        "state",
        "country",
        "iso3",
        "continent",
        "lat",
        "lon",
        "segments",
        "duration_hours",
        "unique_videos",
        "percentage_of_resolved_segments",
    ]
    if resolved.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        resolved.groupby(
            [
                "locality_key",
                "locality",
                "state",
                "country",
                "iso3",
                "continent",
                "lat",
                "lon",
            ],
            dropna=False,
        )
        .agg(
            segments=("video_id", "size"),
            duration_seconds=("duration_seconds", "sum"),
            unique_videos=("video_id", "nunique"),
        )
        .reset_index()
    )
    grouped["duration_hours"] = (grouped["duration_seconds"] / 3600.0).round(2)
    grouped["percentage_of_resolved_segments"] = (
        grouped["segments"] / len(resolved) * 100.0
    ).round(2)
    grouped = grouped.sort_values(
        ["segments", "locality", "country"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    return grouped[columns]


def build_duration_summary(segments: pd.DataFrame) -> pd.DataFrame:
    durations = pd.to_numeric(segments.get("duration_seconds"), errors="coerce").dropna()
    durations = durations.loc[durations >= 0]
    if durations.empty:
        return pd.DataFrame(
            [{
                "count": 0,
                "total_hours": 0.0,
                "mean_seconds": 0.0,
                "median_seconds": 0.0,
                "p25_seconds": 0.0,
                "p75_seconds": 0.0,
                "p90_seconds": 0.0,
                "max_seconds": 0.0,
            }]
        )
    return pd.DataFrame(
        [{
            "count": int(len(durations)),
            "total_hours": round(float(durations.sum()) / 3600.0, 2),
            "mean_seconds": round(float(durations.mean()), 2),
            "median_seconds": round(float(durations.median()), 2),
            "p25_seconds": round(float(durations.quantile(0.25)), 2),
            "p75_seconds": round(float(durations.quantile(0.75)), 2),
            "p90_seconds": round(float(durations.quantile(0.90)), 2),
            "max_seconds": round(float(durations.max()), 2),
        }]
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    state: dict[str, Any],
    segments: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    videos = state.get("videos", {})
    total_videos = len(videos)

    metadata_accepted = 0
    metadata_rejected = 0
    for record in videos.values():
        if not isinstance(record, dict):
            continue
        decision = record.get("text_decision")
        if isinstance(decision, dict) and isinstance(decision.get("include"), bool):
            if decision["include"]:
                metadata_accepted += 1
            else:
                metadata_rejected += 1
        elif record.get("status") == "text_rejected":
            metadata_rejected += 1

    status_counts = {
        str(row["status"]): _safe_int(row["count"])
        for _, row in tables["video_statuses"].iterrows()
    }

    accepted_segments = len(segments)
    complete_videos = status_counts.get("complete", 0)
    source_videos = int(segments["video_id"].nunique()) if accepted_segments else 0
    total_duration_hours = round(
        float(pd.to_numeric(segments["duration_seconds"], errors="coerce").fillna(0).sum())
        / 3600.0,
        2,
    )

    resolved = segments.loc[segments["resolved_location"]].copy()
    resolved_count = len(resolved)
    unresolved_count = accepted_segments - resolved_count
    unique_localities = int(resolved["locality_key"].nunique()) if resolved_count else 0

    country_keys = resolved["iso3"].where(resolved["iso3"].astype(bool), resolved["country"])
    countries = int(country_keys.replace("", pd.NA).dropna().nunique()) if resolved_count else 0
    continents = int(
        resolved["continent"].replace("", pd.NA).dropna().nunique()
    ) if resolved_count else 0

    segment_specific_evidence = int(segments["has_segment_location_evidence"].sum()) if accepted_segments else 0
    metadata_evidence = int(segments["has_video_metadata_location_evidence"].sum()) if accepted_segments else 0
    any_evidence = int(segments["has_any_location_evidence"].sum()) if accepted_segments else 0

    timestamp_any = int(segments["has_timestamp_evidence"].sum()) if accepted_segments else 0
    embedded_any = int(segments["has_embedded_evidence"].sum()) if accepted_segments else 0
    both = int(
        (segments["has_timestamp_evidence"] & segments["has_embedded_evidence"]).sum()
    ) if accepted_segments else 0
    timestamp_only = timestamp_any - both
    embedded_only = embedded_any - both
    neither = accepted_segments - timestamp_only - embedded_only - both

    return {
        "corpus": {
            "video_records_discovered": total_videos,
            "metadata_accepted": metadata_accepted,
            "metadata_rejected": metadata_rejected,
            "complete_videos": complete_videos,
            "unique_source_videos_represented": source_videos,
            "visual_rejected": status_counts.get("visual_rejected", 0),
            "visual_errors": status_counts.get("visual_error", 0),
            "text_errors": status_counts.get("text_error", 0),
            "pipeline_errors": status_counts.get("pipeline_error", 0),
            "accepted_segments": accepted_segments,
            "retained_duration_hours": total_duration_hours,
        },
        "geography": {
            "segments_with_any_geographic_evidence": any_evidence,
            "segments_with_any_geographic_evidence_pct": _percentage(any_evidence, accepted_segments),
            "segments_with_segment_specific_evidence": segment_specific_evidence,
            "segments_with_segment_specific_evidence_pct": _percentage(segment_specific_evidence, accepted_segments),
            "segments_with_video_metadata_evidence": metadata_evidence,
            "segments_with_video_metadata_evidence_pct": _percentage(metadata_evidence, accepted_segments),
            "resolved_segments": resolved_count,
            "resolved_segments_pct": _percentage(resolved_count, accepted_segments),
            "unresolved_segments": unresolved_count,
            "unresolved_segments_pct": _percentage(unresolved_count, accepted_segments),
            "canonical_localities": unique_localities,
            "countries_or_territories": countries,
            "continents": continents,
            "resolved_source_videos": int(resolved["video_id"].nunique()) if resolved_count else 0,
            "resolved_duration_hours": round(float(resolved["duration_seconds"].sum()) / 3600.0, 2) if resolved_count else 0.0,
        },
        "location_provenance": {
            "timestamp_only": timestamp_only,
            "timestamp_only_pct": _percentage(timestamp_only, accepted_segments),
            "embedded_only": embedded_only,
            "embedded_only_pct": _percentage(embedded_only, accepted_segments),
            "both": both,
            "both_pct": _percentage(both, accepted_segments),
            "neither": neither,
            "neither_pct": _percentage(neither, accepted_segments),
            "timestamp_any": timestamp_any,
            "timestamp_any_pct": _percentage(timestamp_any, accepted_segments),
            "embedded_any": embedded_any,
            "embedded_any_pct": _percentage(embedded_any, accepted_segments),
        },
    }


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------


def _table_lines(
    table: pd.DataFrame,
    key_column: str,
    count_column: str = "count",
    percentage_column: str = "percentage",
) -> list[str]:
    lines: list[str] = []
    for _, row in table.iterrows():
        label = row.get("label") or _human_label(row.get(key_column))
        count = _safe_int(row.get(count_column))
        percentage = float(row.get(percentage_column, 0.0) or 0.0)
        lines.append(f"{label:<34} {count:>8,}   {percentage:>6.2f}%")
    return lines


def write_paper_values(
    path: Path,
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> None:
    corpus = summary["corpus"]
    geo = summary["geography"]
    provenance = summary["location_provenance"]

    lines: list[str] = [
        "############################################",
        "### PAPER-FACING WALKING RESULTS VALUES",
        "############################################",
        "",
        "=== Corpus summary ===",
        f"Video records / discovered:      {corpus['video_records_discovered']:,}",
        f"Metadata accepted:                {corpus['metadata_accepted']:,}",
        f"Complete videos:                  {corpus['complete_videos']:,}",
        f"Unique represented source videos: {corpus['unique_source_videos_represented']:,}",
        f"Metadata rejected:                {corpus['metadata_rejected']:,}",
        f"Visual rejected:                  {corpus['visual_rejected']:,}",
        f"Visual errors:                    {corpus['visual_errors']:,}",
        f"Text errors:                      {corpus['text_errors']:,}",
        f"Pipeline errors:                  {corpus['pipeline_errors']:,}",
        f"Accepted segments:                {corpus['accepted_segments']:,}",
        f"Retained duration:                {corpus['retained_duration_hours']:.2f} h",
        "",
        "=== All video statuses ===",
        *_table_lines(tables["video_statuses"], "status"),
        "",
        "=== Walking environment across accepted segments ===",
        *_table_lines(tables["walking_environments"], "walking_environment"),
        "",
        "=== Time of day across accepted segments ===",
        *_table_lines(tables["time_of_day"], "time_of_day"),
        "",
        "=== Segment-specific location provenance ===",
        f"Timestamp / chapter only:         {provenance['timestamp_only']:,}   {provenance['timestamp_only_pct']:.2f}%",
        f"Embedded video text only:         {provenance['embedded_only']:,}   {provenance['embedded_only_pct']:.2f}%",
        f"Both:                             {provenance['both']:,}   {provenance['both_pct']:.2f}%",
        f"Neither:                          {provenance['neither']:,}   {provenance['neither_pct']:.2f}%",
        f"Any timestamp / chapter evidence: {provenance['timestamp_any']:,}   {provenance['timestamp_any_pct']:.2f}%",
        f"Any embedded video text:          {provenance['embedded_any']:,}   {provenance['embedded_any_pct']:.2f}%",
        "",
        "=== Geographic summary ===",
        f"Any geographic evidence:          {geo['segments_with_any_geographic_evidence']:,} ({geo['segments_with_any_geographic_evidence_pct']:.2f}%)",
        f"Segment-specific evidence:        {geo['segments_with_segment_specific_evidence']:,} ({geo['segments_with_segment_specific_evidence_pct']:.2f}%)",
        f"Video metadata evidence:          {geo['segments_with_video_metadata_evidence']:,} ({geo['segments_with_video_metadata_evidence_pct']:.2f}%)",
        f"Resolved segments:                {geo['resolved_segments']:,} ({geo['resolved_segments_pct']:.2f}%)",
        f"Unresolved segments:              {geo['unresolved_segments']:,} ({geo['unresolved_segments_pct']:.2f}%)",
        f"Canonical localities:             {geo['canonical_localities']:,}",
        f"Countries / territories:          {geo['countries_or_territories']:,}",
        f"Continents:                       {geo['continents']:,}",
        f"Resolved source videos:           {geo['resolved_source_videos']:,}",
        f"Resolved duration:                {geo['resolved_duration_hours']:.2f} h",
        "Coordinate semantics:              canonical locality reference coordinates, not exact walking positions",
        "",
        "=== Geocode status across accepted segments ===",
        *_table_lines(tables["geocode_statuses"], "geocode_status"),
        "",
        "=== Resolved segments by continent ===",
    ]

    continent_table = tables["continents"]
    for _, row in continent_table.iterrows():
        lines.append(
            f"{str(row['continent']):<34} {_safe_int(row['segments']):>8,}   {float(row['percentage']):>6.2f}%"
        )

    lines.extend(["", "=== Resolved segments by country / territory ==="])
    country_table = tables["countries"]
    for _, row in country_table.iterrows():
        lines.append(
            f"{str(row['country']):<34} {_safe_int(row['segments']):>8,}   {float(row['percentage']):>6.2f}%"
        )

    lines.extend(["", "=== Top 25 resolved localities ==="])
    localities = tables["localities"].head(25)
    for _, row in localities.iterrows():
        place = ", ".join(
            value
            for value in [
                _clean_text(row.get("locality")),
                _clean_text(row.get("state")),
                _clean_text(row.get("country")),
            ]
            if value
        )
        lines.append(
            f"{place:<62} {_safe_int(row['segments']):>8,}   {float(row['percentage_of_resolved_segments']):>6.2f}%"
        )

    duration = tables["duration_summary"].iloc[0].to_dict()
    lines.extend(
        [
            "",
            "=== Accepted segment duration ===",
            f"Total:                           {float(duration['total_hours']):.2f} h",
            f"Mean:                            {float(duration['mean_seconds']):.2f} s",
            f"Median:                          {float(duration['median_seconds']):.2f} s",
            f"25th percentile:                 {float(duration['p25_seconds']):.2f} s",
            f"75th percentile:                 {float(duration['p75_seconds']):.2f} s",
            f"90th percentile:                 {float(duration['p90_seconds']):.2f} s",
            f"Maximum:                         {float(duration['max_seconds']):.2f} s",
            "",
            "All complete CSV tables are saved under tables/.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class WalkingResultsPlotter:
    MAP_PANEL_COLOURS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    MAP_LAND_COLOUR = "#e6e6e6"
    MAP_COUNTRY_LINE = "#ffffff"
    MAP_COAST_LINE = "#ffffff"
    MAP_OCEAN_COLOUR = "#ffffff"

    def __init__(self, output_dir: Path, publication_dir: Path | None) -> None:
        self.output_dir = output_dir
        self.publication_dir = publication_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.publication_dir:
            self.publication_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _base_layout(
        fig: go.Figure,
        *,
        width: int = 1000,
        height: int = 620,
    ) -> None:
        fig.update_layout(
            template="plotly_white",
            width=width,
            height=height,
            font=dict(size=15),
            margin=dict(l=30, r=30, t=80, b=30),
            legend_title_text="",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    def _apply_geo_style(self, fig: go.Figure, geo_name: str = "geo") -> None:
        getattr(fig.layout, geo_name).update(
            scope="world",
            projection_type="natural earth",
            projection_scale=1.0,
            center=dict(lat=0, lon=0),
            showframe=False,
            showland=True,
            landcolor=self.MAP_LAND_COLOUR,
            showocean=True,
            oceancolor=self.MAP_OCEAN_COLOUR,
            showcountries=True,
            countrycolor=self.MAP_COUNTRY_LINE,
            countrywidth=0.6,
            showcoastlines=True,
            coastlinecolor=self.MAP_COAST_LINE,
            coastlinewidth=0.4,
            bgcolor="white",
            lataxis=dict(showgrid=False),
            lonaxis=dict(showgrid=False),
        )

    def save(
        self,
        fig: go.Figure,
        name: str,
        *,
        width: int = 1000,
        height: int = 620,
    ) -> None:
        self._base_layout(fig, width=width, height=height)

        html_path = self.output_dir / f"{name}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        produced = [html_path]

        for suffix in ("png", "pdf"):
            path = self.output_dir / f"{name}.{suffix}"
            try:
                fig.write_image(
                    path,
                    width=width,
                    height=height,
                    scale=2 if suffix == "png" else 1,
                )
                produced.append(path)
            except Exception as exc:
                print(f"Warning: could not write {path.name}: {exc}")

        if self.publication_dir:
            for path in produced:
                shutil.copy2(path, self.publication_dir / path.name)

    def bar_distribution(
        self,
        table: pd.DataFrame,
        *,
        label_col: str,
        count_col: str,
        title: str,
        x_title: str,
        name: str,
        horizontal: bool = False,
    ) -> None:
        if table.empty:
            return
        data = table.copy()
        if horizontal:
            data = data.iloc[::-1]
            fig = go.Figure(
                go.Bar(
                    x=data[count_col],
                    y=data[label_col],
                    orientation="h",
                    text=[f"{int(v):,}" for v in data[count_col]],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig.update_xaxes(title=x_title)
        else:
            fig = go.Figure(
                go.Bar(
                    x=data[label_col],
                    y=data[count_col],
                    text=[f"{int(v):,}" for v in data[count_col]],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig.update_yaxes(title=x_title)
        fig.update_layout(title=title)
        self.save(fig, name)

    def duration_histogram(self, segments: pd.DataFrame) -> None:
        if segments.empty:
            return
        durations = pd.to_numeric(segments["duration_seconds"], errors="coerce").dropna()
        durations = durations.loc[durations > 0] / 60.0
        if durations.empty:
            return
        fig = go.Figure(
            go.Histogram(
                x=durations,
                nbinsx=40,
            )
        )
        fig.update_layout(title="Accepted walking segment duration")
        fig.update_xaxes(title="Segment duration (minutes)")
        fig.update_yaxes(title="Segments")
        self.save(fig, "fig_walking_segment_duration")

    @staticmethod
    def _aggregate_geo(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame()
        return (
            data.groupby(
                [
                    "locality_key",
                    "locality",
                    "state",
                    "country",
                    "lat",
                    "lon",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="segments")
            .sort_values("segments", ascending=False, ignore_index=True)
        )

    @staticmethod
    def _bubble_sizes(
        counts: pd.Series,
        *,
        global_max: float,
        minimum: float = 4.0,
        maximum: float = 18.0,
    ) -> list[float]:
        denominator = max(float(global_max), 1.0)
        return [
            minimum
            + (maximum - minimum)
            * math.sqrt(max(float(value), 0.0) / denominator)
            for value in counts
        ]

    def world_map_all(self, resolved: pd.DataFrame) -> None:
        geo = self._aggregate_geo(resolved)
        if geo.empty:
            return
        sizes = self._bubble_sizes(geo["segments"], global_max=float(geo["segments"].max()))
        hover = [
            "<br>".join(
                filter(
                    None,
                    [
                        _clean_text(row.locality),
                        _clean_text(row.state),
                        _clean_text(row.country),
                        f"Segments: {int(row.segments):,}",
                    ],
                )
            )
            for row in geo.itertuples(index=False)
        ]
        fig = go.Figure(
            go.Scattergeo(
                lat=geo["lat"],
                lon=geo["lon"],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=self.MAP_PANEL_COLOURS[0],
                    opacity=0.78,
                    line=dict(color="#ffffff", width=0.7),
                ),
                text=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        self._apply_geo_style(fig, "geo")
        fig.update_layout(title=None)
        self.save(fig, "fig_world_walking_all", width=1400, height=780)

    def world_map_environments(
        self,
        segments: pd.DataFrame,
        *,
        top_n: int,
    ) -> None:
        if segments.empty:
            return
        resolved = segments.loc[segments["resolved_location"]].copy()
        if resolved.empty:
            return

        overall_counts = (
            segments["walking_environment"]
            .value_counts()
            .drop(labels=["unknown", "not_applicable"], errors="ignore")
        )
        categories = overall_counts.head(max(1, top_n)).index.tolist()
        if not categories:
            return

        cols = 2 if len(categories) > 1 else 1
        rows = math.ceil(len(categories) / cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "geo"} for _ in range(cols)] for _ in range(rows)],
            vertical_spacing=0.06,
            horizontal_spacing=0.04,
        )

        grouped_panels: list[tuple[str, pd.DataFrame]] = []
        global_max = 1.0
        for category in categories:
            panel = self._aggregate_geo(
                resolved.loc[resolved["walking_environment"] == category]
            )
            grouped_panels.append((category, panel))
            if not panel.empty:
                global_max = max(global_max, float(panel["segments"].max()))

        for index, (category, panel) in enumerate(grouped_panels):
            row = index // cols + 1
            col = index % cols + 1
            if panel.empty:
                continue
            sizes = self._bubble_sizes(panel["segments"], global_max=global_max)
            hover = [
                "<br>".join(
                    filter(
                        None,
                        [
                            _clean_text(item.locality),
                            _clean_text(item.state),
                            _clean_text(item.country),
                            f"Segments: {int(item.segments):,}",
                        ],
                    )
                )
                for item in panel.itertuples(index=False)
            ]
            colour = self.MAP_PANEL_COLOURS[index % len(self.MAP_PANEL_COLOURS)]
            fig.add_trace(
                go.Scattergeo(
                    lat=panel["lat"],
                    lon=panel["lon"],
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=colour,
                        opacity=0.82,
                        line=dict(color="#ffffff", width=0.6),
                    ),
                    text=hover,
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        for index in range(1, rows * cols + 1):
            geo_name = "geo" if index == 1 else f"geo{index}"
            if hasattr(fig.layout, geo_name):
                self._apply_geo_style(fig, geo_name)

        fig.update_layout(title=None)
        self.save(
            fig,
            "fig_world_walking_environment",
            width=1400,
            height=max(900, rows * 430),
        )


def generate_figures(
    segments: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    publication_dir: Path,
    *,
    top_countries: int,
    top_environments: int,
) -> None:
    plotter = WalkingResultsPlotter(output_dir, publication_dir)

    environments = tables["walking_environments"].copy()
    if not environments.empty:
        plotter.bar_distribution(
            environments,
            label_col="label",
            count_col="count",
            title="Walking environments in accepted segments",
            x_title="Accepted segments",
            name="fig_walking_environment_distribution",
            horizontal=True,
        )

    time_table = tables["time_of_day"].copy()
    if not time_table.empty:
        plotter.bar_distribution(
            time_table,
            label_col="label",
            count_col="count",
            title="Time of day in accepted walking segments",
            x_title="Accepted segments",
            name="fig_walking_time_of_day",
        )

    location_source = tables["location_sources"].copy()
    if not location_source.empty:
        plotter.bar_distribution(
            location_source,
            label_col="label",
            count_col="count",
            title="Segment-specific geographic evidence provenance",
            x_title="Accepted segments",
            name="fig_walking_location_provenance",
        )

    countries = tables["countries"].head(max(1, top_countries)).copy()
    if not countries.empty:
        countries["label"] = countries["country"]
        plotter.bar_distribution(
            countries,
            label_col="label",
            count_col="segments",
            title=f"Top {len(countries)} countries by resolved walking segments",
            x_title="Resolved segments",
            name="fig_walking_top_countries",
            horizontal=True,
        )

    plotter.duration_histogram(segments)
    resolved = segments.loc[segments["resolved_location"]].copy()
    plotter.world_map_all(resolved)
    plotter.world_map_environments(segments, top_n=top_environments)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    state = load_state(args.state)
    segments = extract_segments(state)
    resolved = segments.loc[segments["resolved_location"]].copy()

    tables: dict[str, pd.DataFrame] = {
        "video_statuses": build_video_status_table(state),
        "walking_environments": build_environment_table(segments),
        "time_of_day": build_time_table(segments),
        "location_sources": build_location_source_table(segments),
        "location_evidence": build_location_evidence_table(segments),
        "geocode_statuses": build_geocode_status_table(segments),
        "countries": build_country_table(resolved),
        "continents": build_continent_table(resolved),
        "localities": build_locality_table(resolved),
        "duration_summary": build_duration_summary(segments),
    }

    summary = build_summary(state, segments, tables)

    output_dir: Path = args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)

    # Segment-level table is useful for checking manuscript values and for
    # producing additional exploratory figures without re-reading state.json.
    segments.to_csv(table_dir / "accepted_segments.csv", index=False)

    with (output_dir / "paper_values.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    write_paper_values(output_dir / "paper_values.txt", summary, tables)

    if not args.no_plots:
        generate_figures(
            segments,
            tables,
            figure_dir,
            args.publication_dir,
            top_countries=max(1, args.top_countries),
            top_environments=max(1, args.map_top_environments),
        )

    corpus = summary["corpus"]
    geo = summary["geography"]
    print("Walking corpus analysis complete")
    print(f"  Video records:       {corpus['video_records_discovered']:,}")
    print(f"  Complete videos:     {corpus['complete_videos']:,}")
    print(f"  Accepted segments:   {corpus['accepted_segments']:,}")
    print(f"  Retained duration:   {corpus['retained_duration_hours']:.2f} h")
    print(f"  Resolved segments:   {geo['resolved_segments']:,}")
    print(f"  Canonical localities:{geo['canonical_localities']:>9,}")
    print(f"  Countries:           {geo['countries_or_territories']:,}")
    print(f"  Continents:          {geo['continents']:,}")
    print(f"  Paper values:        {output_dir / 'paper_values.txt'}")
    print(f"  Tables:              {table_dir}")
    if not args.no_plots:
        print(f"  Figures:             {figure_dir}")
        print(f"  Publication figures: {args.publication_dir}")


if __name__ == "__main__":
    main()
