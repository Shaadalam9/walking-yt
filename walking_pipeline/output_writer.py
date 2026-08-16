"""Locality aggregation and final CSV writing."""

from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from . import settings
from .shared import normalise_string_list, optional_text


OUTPUT_WRITER_SCHEMA_VERSION = "walking_incremental_location_csv_v3"
_LOCATION_OUTPUT_COLUMNS = (
    "walking_environment",
    "timestamp_labels",
    "embedded_location_text",
    "location_source",
)


def resolved_output_columns() -> List[str]:
    """Return a compatible CSV schema even with an older settings file."""
    columns = [
        column
        for column in settings.OUTPUT_COLUMNS
        if column != "vehicle_type"
    ]
    insertion_index = (
        columns.index("start_time")
        if "start_time" in columns
        else len(columns)
    )
    for column in _LOCATION_OUTPUT_COLUMNS:
        if column in columns:
            continue
        columns.insert(insertion_index, column)
        insertion_index += 1
    return columns


def location_key(
    location: Dict[str, Any], video_id: Optional[str] = None
) -> str:
    locality = optional_text(location.get("locality"))
    lat = location.get("lat")
    lon = location.get("lon")
    if not locality and (lat is None or lon is None):
        return json.dumps(
            ["unresolved_video", video_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    values = [
        locality,
        optional_text(location.get("state")),
        optional_text(location.get("country")),
        lat,
        lon,
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def format_upload_date(value: Any) -> Optional[int]:
    text = optional_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return int(parsed.strftime("%d%m%Y"))
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text[:10])
            return int(parsed_date.strftime("%d%m%Y"))
        except ValueError:
            return None


def _bracket_text(value: Any) -> str:
    """Return one readable scalar without list delimiter ambiguity."""
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace(",", ";")
        .strip()
    )


def bracket_cell(value: Any) -> str:
    """Encode aligned CSV values using non-JSON square brackets.

    Timestamp dictionaries use the compact order
    [timestamp_seconds,timestamp_text,label]. Text commas become semicolons
    so commas remain unambiguous list separators without quoting each item.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(bracket_cell(item) for item in value) + "]"
    if isinstance(value, dict):
        timestamp_keys = (
            "timestamp_seconds",
            "timestamp_text",
            "label",
        )
        if any(key in value for key in timestamp_keys):
            items = [value.get(key) for key in timestamp_keys]
        else:
            items = [
                f"{_bracket_text(key)}={bracket_cell(item)}"
                for key, item in value.items()
            ]
        return "[" + ",".join(bracket_cell(item) for item in items) + "]"
    return _bracket_text(value)


def scalar_cell(value: Any) -> Any:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ").replace(
            '"', "'"
        )
    return value


def write_output_csv(state: Dict[str, Any]) -> None:
    grouped: Dict[str, Dict[str, Any]] = {}
    locality_ids = state.setdefault("locality_ids", {})
    existing_ids = [
        int(value)
        for value in locality_ids.values()
        if isinstance(value, int)
    ]
    next_id = max([settings.FIRST_LOCALITY_ID - 1, *existing_ids]) + 1

    for video_id, record in state.get("videos", {}).items():
        if record.get("status") != "complete":
            continue
        location = _record_location(record)
        key = location_key(location, video_id)
        if key not in locality_ids:
            locality_ids[key] = next_id
            next_id += 1
        if key not in grouped:
            grouped[key] = _new_group(locality_ids[key], location)
        _append_video(grouped[key], video_id, record)

    _write_groups(grouped)


def _record_location(record: Dict[str, Any]) -> Dict[str, Any]:
    location = record.get("location")
    if not isinstance(location, dict):
        location = {}
    decision = record.get("text_decision")
    if not isinstance(decision, dict):
        decision = {}
    return {
        "locality": (
            location.get("locality") or decision.get("locality")
        ),
        "locality_aka": location.get("locality_aka", []),
        "state": location.get("state") or decision.get("state"),
        "country": (
            location.get("country") or decision.get("country")
        ),
        "iso3": location.get("iso3"),
        "continent": location.get("continent"),
        "lat": location.get("lat"),
        "lon": location.get("lon"),
    }


def _new_group(
    locality_id: int, location: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "id": locality_id,
        "locality": location.get("locality"),
        "locality_aka": normalise_string_list(
            location.get("locality_aka")
        ),
        "state": location.get("state"),
        "country": location.get("country"),
        "iso3": location.get("iso3"),
        "continent": location.get("continent"),
        "lat": location.get("lat"),
        "lon": location.get("lon"),
        "videos": [],
        "time_of_day": [],
        "walking_environment": [],
        "timestamp_labels": [],
        "embedded_location_text": [],
        "location_source": [],
        "start_time": [],
        "end_time": [],
        "upload_date": [],
        "channel": [],
    }


def _append_video(
    group: Dict[str, Any],
    video_id: str,
    record: Dict[str, Any],
) -> None:
    location = _record_location(record)
    for alternative in normalise_string_list(location.get("locality_aka")):
        if alternative not in group["locality_aka"]:
            group["locality_aka"].append(alternative)

    segments = record.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    group["videos"].append(video_id)
    group["time_of_day"].append(
        [
            int(segment.get("time_of_day", -1))
            for segment in segments
            if isinstance(segment, dict)
        ]
    )
    group["walking_environment"].append(
        [
            str(segment.get("walking_environment", "unknown"))
            for segment in segments
            if isinstance(segment, dict)
        ]
    )
    group["timestamp_labels"].append(
        [
            segment.get("timestamp_labels", [])
            for segment in segments
            if isinstance(segment, dict)
        ]
    )
    group["embedded_location_text"].append(
        [
            segment.get("embedded_location_text", [])
            for segment in segments
            if isinstance(segment, dict)
        ]
    )
    group["location_source"].append(
        [
            str(segment.get("location_source", "none"))
            for segment in segments
            if isinstance(segment, dict)
        ]
    )
    group["start_time"].append(
        [
            int(segment.get("start_time", 0))
            for segment in segments
            if isinstance(segment, dict)
        ]
    )
    group["end_time"].append(
        [
            int(segment.get("end_time", 0))
            for segment in segments
            if isinstance(segment, dict)
        ]
    )
    metadata = record.get("metadata", {})
    group["upload_date"].append(
        format_upload_date(metadata.get("upload_date"))
    )
    group["channel"].append(metadata.get("channel"))


def _write_groups(grouped: Dict[str, Dict[str, Any]]) -> None:
    settings.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = settings.OUTPUT_CSV.with_suffix(
        settings.OUTPUT_CSV.suffix + f".tmp.{os.getpid()}"
    )
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=resolved_output_columns()
        )
        writer.writeheader()
        for group in sorted(grouped.values(), key=lambda row: row["id"]):
            writer.writerow(_serialise_group(group))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, settings.OUTPUT_CSV)


def _serialise_group(group: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": group["id"],
        "locality": scalar_cell(group["locality"]),
        "locality_aka": bracket_cell(group["locality_aka"]),
        "state": scalar_cell(group["state"]),
        "country": scalar_cell(group["country"]),
        "iso3": scalar_cell(group["iso3"]),
        "continent": scalar_cell(group["continent"]),
        "lat": scalar_cell(group["lat"]),
        "lon": scalar_cell(group["lon"]),
        "videos": bracket_cell(group["videos"]),
        "time_of_day": bracket_cell(group["time_of_day"]),
        "walking_environment": bracket_cell(
            group["walking_environment"]
        ),
        "timestamp_labels": bracket_cell(group["timestamp_labels"]),
        "embedded_location_text": bracket_cell(
            group["embedded_location_text"]
        ),
        "location_source": bracket_cell(group["location_source"]),
        "start_time": bracket_cell(group["start_time"]),
        "end_time": bracket_cell(group["end_time"]),
        "upload_date": bracket_cell(group["upload_date"]),
        "channel": bracket_cell(group["channel"]),
    }