"""Environment based configuration for the walking video pipeline.

The module deliberately validates environment values at import time. This
makes misspelled or malformed Run:ai settings fail before large models are
loaded or data processing begins.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_NONE_VALUES = {"", "none", "null", "unlimited", "all"}
_VALID_PIPELINE_MODES = {"auto", "sequential", "overlap"}


def env_bool(name: str, default: bool) -> bool:
    """Read a strict Boolean environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    allowed = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(
        f"{name} must be one of: {allowed}. Received: {raw_value!r}"
    )


def env_int(
    name: str,
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Read and range check an integer environment variable."""
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer. Received: {raw_value!r}"
        ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}. Received: {value}")
    return value


def env_optional_int(
    name: str,
    default: Optional[int],
    *,
    minimum: int = 0,
) -> Optional[int]:
    """Read an optional integer, accepting none or unlimited as no limit."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    value = raw_value.strip().lower()
    if value in _NONE_VALUES:
        return None
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer or one of: none, unlimited. "
            f"Received: {raw_value!r}"
        ) from exc
    if number < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}. Received: {number}"
        )
    return number


def env_float(
    name: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read and range check a floating point environment variable."""
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number. Received: {raw_value!r}"
        ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}. Received: {value}")
    return value


def env_device(name: str, default: str) -> str:
    """Read a torch device, accepting a bare CUDA index such as 0."""
    value = os.environ.get(name, default).strip().lower()
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if value.isdigit():
        return f"cuda:{value}"
    if value == "cuda":
        return "cuda:0"
    return value


def _data_child_path(name: str, default: str) -> Path:
    value = Path(os.environ.get(name, default)).expanduser()
    return value if value.is_absolute() else DATA_DIR / value


SPIKE1_MODE = env_bool("SPIKE1_MODE", False)
SPIKE1_REQUIRE_GPU = env_bool("SPIKE1_REQUIRE_GPU", SPIKE1_MODE)
SPIKE1_REQUIRE_PERSISTENT_STORAGE = env_bool(
    "SPIKE1_REQUIRE_PERSISTENT_STORAGE", SPIKE1_MODE
)

DATA_DIR = Path(os.environ.get("WALK_DATA_DIR", "data")).expanduser()
STATE_JSON = _data_child_path(
    "WALK_STATE_JSON", "walking_pipeline_state.json"
)
OUTPUT_CSV = _data_child_path("WALK_OUTPUT_CSV", "walking_segments.csv")
VIDEO_DIR = _data_child_path("WALK_VIDEO_DIR", "walking_videos")
GEOCODE_CACHE_JSON = _data_child_path(
    "WALK_GEOCODE_CACHE", "walking_geocode_cache.json"
)

_default_hf_home = (
    DATA_DIR.parent / "huggingface"
    if SPIKE1_MODE
    else Path.home() / ".cache" / "huggingface"
)
HF_HOME = Path(os.environ.get("HF_HOME", str(_default_hf_home))).expanduser()
HF_HUB_CACHE = Path(
    os.environ.get("HF_HUB_CACHE", str(HF_HOME / "hub"))
).expanduser()

# Transformers reads these variables itself. Setting defaults here ensures
# model downloads are kept on persistent storage in Spike 1 mode.
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_WALKING_QUERIES = [
    "walking tour",
    "city walk",
    "walk with me",
    "virtual walk",
    "4k walking tour",
    "POV walking",
    "street walk",
    "nature walk",
    "rain walk",
    "night walk",
]

MAX_PAGES_PER_QUERY = env_int("MAX_PAGES_PER_QUERY", 2, minimum=1)
RESULTS_PER_PAGE = env_int(
    "RESULTS_PER_PAGE", 50, minimum=1, maximum=50
)
MAX_NEW_CANDIDATES: Optional[int] = env_optional_int(
    "MAX_NEW_CANDIDATES", 200
)
MAX_VIDEOS_PER_RUN: Optional[int] = env_optional_int(
    "MAX_VIDEOS_PER_RUN", 20
)
PUBLISHED_AFTER = os.environ.get("PUBLISHED_AFTER") or None
PUBLISHED_BEFORE = os.environ.get("PUBLISHED_BEFORE") or None

TEXT_MODEL_NAME = os.environ.get(
    "TEXT_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
).strip()
TEXT_LOAD_IN_4BIT = env_bool("TEXT_LLM_4BIT", False)
TEXT_MAX_NEW_TOKENS = env_int("TEXT_MAX_NEW_TOKENS", 384, minimum=1)
MIN_TEXT_CONFIDENCE = env_float(
    "MIN_TEXT_CONFIDENCE", 0.60, minimum=0.0, maximum=1.0
)

VLM_MODEL_NAME = os.environ.get(
    "INTERNVL_MODEL", "OpenGVLab/InternVL3_5-14B-HF"
).strip()
VLM_LOAD_IN_4BIT = env_bool("INTERNVL_4BIT", False)
TEMPORAL_SAMPLES = env_int("TEMPORAL_SAMPLES", 6, minimum=1)
SAMPLE_SECONDS = env_int("SAMPLE_SECONDS", 6, minimum=1)
FRAMES_PER_SAMPLE = env_int("FRAMES_PER_SAMPLE", 6, minimum=1)
VLM_MAX_NEW_TOKENS = env_int("VLM_MAX_NEW_TOKENS", 320, minimum=1)
MIN_INCLUDED_SAMPLE_RATIO = env_float(
    "MIN_INCLUDED_SAMPLE_RATIO", 0.70, minimum=0.0, maximum=1.0
)
MIN_VISUAL_CONFIDENCE = env_float(
    "MIN_VISUAL_CONFIDENCE", 0.75, minimum=0.0, maximum=1.0
)

# Application stage overlap is separate from model tensor parallelism.
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "auto").strip().lower()
if PIPELINE_MODE not in _VALID_PIPELINE_MODES:
    allowed_modes = ", ".join(sorted(_VALID_PIPELINE_MODES))
    raise ValueError(
        f"PIPELINE_MODE must be one of: {allowed_modes}. "
        f"Received: {PIPELINE_MODE!r}"
    )

# Run:ai presents allocated GPUs as logical cuda:0, cuda:1, and so on inside
# the container, even when their physical host identifiers are different.
SEQUENTIAL_DEVICE = env_device("SEQUENTIAL_DEVICE", "cuda:0")
TEXT_DEVICE = env_device("TEXT_DEVICE", "cuda:0")
VLM_DEVICE = env_device("VLM_DEVICE", "cuda:1")
TEXT_MAX_OUTSTANDING = env_int(
    "TEXT_MAX_OUTSTANDING", 16, minimum=1
)
# Two outstanding visual jobs means one is running and one is waiting.
VISUAL_MAX_OUTSTANDING = env_int(
    "VISUAL_MAX_OUTSTANDING", 2, minimum=1
)
WORKER_POLL_SECONDS = env_float(
    "WORKER_POLL_SECONDS", 0.10, minimum=0.01
)
WORKER_START_TIMEOUT_SECONDS = env_int(
    "WORKER_START_TIMEOUT_SECONDS", 1800, minimum=30
)

VIDEO_FORMAT = os.environ.get(
    "WALK_VIDEO_FORMAT",
    (
        "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/"
        "best[height<=480][ext=mp4]/best[height<=480]/best"
    ),
)
KEEP_REJECTED_VIDEOS = env_bool("KEEP_REJECTED_VIDEOS", False)

SCENE_THRESHOLD = env_float(
    "SCENE_THRESHOLD", 0.45, minimum=0.0, maximum=1.0
)
MERGE_NEARBY_SEC = env_float("MERGE_NEARBY_SEC", 2.0, minimum=0.0)
IGNORE_FIRST_SEC = env_float("IGNORE_FIRST_SEC", 0.0, minimum=0.0)
IGNORE_LAST_SEC = env_float("IGNORE_LAST_SEC", 0.0, minimum=0.0)
CUT_DETECTION_TIMEOUT = env_int(
    "CUT_DETECTION_TIMEOUT", 7200, minimum=1
)

ENABLE_GEOCODING = env_bool("ENABLE_GEOCODING", True)
GEOCODER_USER_AGENT = os.environ.get(
    "GEOCODER_USER_AGENT", "walking-video-research-pipeline/1.0"
).strip()
GEOCODER_DELAY_SECONDS = env_float(
    "GEOCODER_DELAY_SECONDS", 1.1, minimum=0.0
)

TIME_OF_DAY_CODES = {
    "day": 0,
    "night": 1,
    "dawn_dusk": 2,
    "unknown": -1,
}
PEDESTRIAN_VEHICLE_TYPE = 0
FIRST_LOCALITY_ID = 1

OUTPUT_COLUMNS = [
    "id",
    "locality",
    "locality_aka",
    "state",
    "country",
    "iso3",
    "continent",
    "lat",
    "lon",
    "videos",
    "time_of_day",
    "start_time",
    "end_time",
    "vehicle_type",
    "upload_date",
    "channel",
]

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
