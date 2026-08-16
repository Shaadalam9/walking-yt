"""Validated JSON configuration for the walking video pipeline.

All user controlled pipeline values come from root entries in the ``config``
file through ``common.get_configs``. The module validates values at import
time so configuration errors are reported before models are loaded or videos
are downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from common import get_configs, root_dir


_VALID_PIPELINE_MODES = {"auto", "sequential", "overlap"}
_VALID_VISUAL_MODEL_BACKENDS = {"cosmos3", "internvl"}
_VALID_CUT_DETECTION_BACKENDS = {
    "auto",
    "ffmpeg_cpu",
    "ffmpeg_cuda",
}

# Printed during installation checks so mismatched revisions are obvious.
SETTINGS_SCHEMA_VERSION = "walking_single_gpu_throughput_v20"


def _config_value(name: str) -> Any:
    try:
        return get_configs(name)
    except KeyError as exc:
        raise ValueError(
            f"Missing configuration value in config: {name}"
        ) from exc


def config_bool(name: str) -> bool:
    """Read a JSON Boolean configuration value."""
    value = _config_value(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false. Received: {value!r}")
    return value


def config_int(
    name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Read and range check a JSON integer configuration value."""
    value = _config_value(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer. Received: {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}. Received: {value}")
    return value


def config_optional_int(
    name: str,
    *,
    minimum: int = 0,
) -> Optional[int]:
    """Read an integer or null from the JSON configuration."""
    value = _config_value(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer or null. Received: {value!r}"
        )
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {value}")
    return value


def config_float(
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read and range check a JSON number configuration value."""
    value = _config_value(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number. Received: {value!r}")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}. Received: {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}. Received: {number}")
    return number


def config_text(name: str, *, allow_empty: bool = False) -> str:
    """Read and trim a JSON string configuration value."""
    value = _config_value(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string. Received: {value!r}")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    return text


def config_optional_text(name: str) -> Optional[str]:
    """Read a nonempty JSON string or null."""
    value = _config_value(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string or null. Received: {value!r}"
        )
    return value.strip() or None


def config_device(name: str) -> str:
    """Read a torch device, accepting a bare CUDA index such as 0."""
    value = config_text(name).lower()
    if value.isdigit():
        return f"cuda:{value}"
    if value == "cuda":
        return "cuda:0"
    return value


def config_text_list(name: str) -> List[str]:
    """Read a nonempty list of unique nonempty strings."""
    value = _config_value(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list. Received: {value!r}")

    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Every {name} entry must be a nonempty string. "
                f"Received: {item!r}"
            )
        text = item.strip()
        if text not in result:
            result.append(text)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _data_child_path(name: str) -> Path:
    value = Path(config_text(name)).expanduser()
    return value if value.is_absolute() else DATA_DIR / value


def _set_process_environment(name: str, value: Optional[str]) -> None:
    """Apply config backed values required by third party libraries."""
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


SPIKE1_MODE = config_bool("SPIKE1_MODE")
SPIKE1_REQUIRE_GPU = config_bool("SPIKE1_REQUIRE_GPU")
SPIKE1_REQUIRE_PERSISTENT_STORAGE = config_bool(
    "SPIKE1_REQUIRE_PERSISTENT_STORAGE"
)

DATA_DIR = Path(config_text("WALK_DATA_DIR")).expanduser()
STATE_JSON = _data_child_path("WALK_STATE_JSON")
OUTPUT_CSV = _data_child_path("WALK_OUTPUT_CSV")
VIDEO_DIR = _data_child_path("WALK_VIDEO_DIR")
VIDEO_TMP_DIR = _data_child_path("WALK_VIDEO_TMP_DIR")
GEOCODE_CACHE_JSON = _data_child_path("WALK_GEOCODE_CACHE")

_resolved_video_dir = VIDEO_DIR.resolve()
_resolved_video_tmp_dir = VIDEO_TMP_DIR.resolve()
if (
    VIDEO_TMP_DIR.name != ".tmp"
    or _resolved_video_dir not in _resolved_video_tmp_dir.parents
):
    raise ValueError(
        "WALK_VIDEO_TMP_DIR must name a .tmp directory inside "
        "WALK_VIDEO_DIR."
    )

_default_hf_home = (
    DATA_DIR.parent / "huggingface"
    if SPIKE1_MODE
    else Path.home() / ".cache" / "huggingface"
)
_configured_hf_home = config_optional_text("HF_HOME")
HF_HOME = (
    Path(_configured_hf_home).expanduser()
    if _configured_hf_home
    else _default_hf_home
)
_configured_hf_hub_cache = config_optional_text("HF_HUB_CACHE")
HF_HUB_CACHE = (
    Path(_configured_hf_hub_cache).expanduser()
    if _configured_hf_hub_cache
    else HF_HOME / "hub"
)

# These libraries still consume process environment variables internally, but
# their values are controlled exclusively by the JSON config file.
_set_process_environment("HF_HOME", str(HF_HOME))
_set_process_environment("HF_HUB_CACHE", str(HF_HUB_CACHE))
_set_process_environment(
    "TOKENIZERS_PARALLELISM",
    "true" if config_bool("TOKENIZERS_PARALLELISM") else "false",
)
_set_process_environment(
    "HF_XET_HIGH_PERFORMANCE",
    "1" if config_bool("HF_XET_HIGH_PERFORMANCE") else "0",
)
_set_process_environment(
    "PYTORCH_CUDA_ALLOC_CONF",
    config_optional_text("PYTORCH_CUDA_ALLOC_CONF"),
)

DEFAULT_WALKING_QUERIES = config_text_list("WALKING_QUERIES")
MAX_PAGES_PER_QUERY = config_int("MAX_PAGES_PER_QUERY", minimum=1)
RESULTS_PER_PAGE = config_int(
    "RESULTS_PER_PAGE", minimum=1, maximum=50
)
MAX_NEW_CANDIDATES = config_optional_int("MAX_NEW_CANDIDATES")
MAX_VIDEOS_PER_RUN = config_optional_int("MAX_VIDEOS_PER_RUN")
CONTINUOUS_BATCH_MODE = config_bool("CONTINUOUS_BATCH_MODE")
CONTINUOUS_BATCH_PAUSE_SECONDS = config_int(
    "CONTINUOUS_BATCH_PAUSE_SECONDS", minimum=0
)
CONTINUOUS_IDLE_PAUSE_SECONDS = config_int(
    "CONTINUOUS_IDLE_PAUSE_SECONDS", minimum=1
)
VIDEO_DOWNLOAD_QUEUE_SIZE = config_int(
    "VIDEO_DOWNLOAD_QUEUE_SIZE", minimum=1
)
VIDEO_DOWNLOAD_FAILURE_ALLOWANCE_PER_RUN = config_int(
    "VIDEO_DOWNLOAD_FAILURE_ALLOWANCE_PER_RUN", minimum=0
)
VIDEO_DOWNLOAD_MAX_CONSECUTIVE_FAILURES = config_int(
    "VIDEO_DOWNLOAD_MAX_CONSECUTIVE_FAILURES", minimum=1
)
VIDEO_DOWNLOAD_STOP_ON_AUTH_ERROR = config_bool(
    "VIDEO_DOWNLOAD_STOP_ON_AUTH_ERROR"
)
DELETE_VIDEO_AFTER_PROCESSING = config_bool(
    "DELETE_VIDEO_AFTER_PROCESSING"
)
MIN_VIDEO_DURATION_SECONDS = config_int(
    "MIN_VIDEO_DURATION_SECONDS", minimum=1
)
PUBLISHED_AFTER = config_optional_text("PUBLISHED_AFTER")
PUBLISHED_BEFORE = config_optional_text("PUBLISHED_BEFORE")

TEXT_MODEL_NAME = config_text("TEXT_LLM_MODEL")
TEXT_LOAD_IN_4BIT = config_bool("TEXT_LLM_4BIT")
TEXT_MAX_NEW_TOKENS = config_int("TEXT_MAX_NEW_TOKENS", minimum=1)
MIN_TEXT_CONFIDENCE = config_float(
    "MIN_TEXT_CONFIDENCE", minimum=0.0, maximum=1.0
)

VLM_MODEL_NAME = config_text("INTERNVL_MODEL")
VLM_LOAD_IN_4BIT = config_bool("INTERNVL_4BIT")
FRAMES_PER_SAMPLE = config_int("FRAMES_PER_SAMPLE", minimum=1)
VLM_MAX_NEW_TOKENS = config_int("VLM_MAX_NEW_TOKENS", minimum=1)
VISUAL_MODEL_BACKEND = config_text("VISUAL_MODEL_BACKEND").lower()
if VISUAL_MODEL_BACKEND not in _VALID_VISUAL_MODEL_BACKENDS:
    allowed_backends = ", ".join(sorted(_VALID_VISUAL_MODEL_BACKENDS))
    raise ValueError(
        f"VISUAL_MODEL_BACKEND must be one of: {allowed_backends}. "
        f"Received: {VISUAL_MODEL_BACKEND!r}"
    )

COSMOS3_MODEL_NAME = config_text("COSMOS3_MODEL")
COSMOS3_LOAD_IN_4BIT = config_bool("COSMOS3_4BIT")
COSMOS3_MAX_NEW_TOKENS = config_int(
    "COSMOS3_MAX_NEW_TOKENS", minimum=1
)
COSMOS3_BATCH_SIZE = config_int(
    "COSMOS3_BATCH_SIZE", minimum=1, maximum=8
)
CLIP_PREPARE_WORKERS = config_int(
    "CLIP_PREPARE_WORKERS", minimum=1, maximum=8
)
COSMOS3_FAST_MODE = config_bool("COSMOS3_FAST_MODE")
COSMOS3_INTRO_SEARCH_SECONDS = config_int(
    "COSMOS3_INTRO_SEARCH_SECONDS", minimum=15
)
COSMOS3_INTRO_FPS = config_float(
    "COSMOS3_INTRO_FPS", minimum=0.25, maximum=4.0
)
COSMOS3_MIN_CONTENT_START_CONFIDENCE = config_float(
    "COSMOS3_MIN_CONTENT_START_CONFIDENCE",
    minimum=0.0,
    maximum=1.0,
)
COSMOS3_FRAMES_PER_SAMPLE = config_int(
    "COSMOS3_FRAMES_PER_SAMPLE", minimum=4
)
COSMOS3_REVIEW_FPS = config_float(
    "COSMOS3_REVIEW_FPS", minimum=0.5, maximum=4.0
)
COSMOS3_LONG_WINDOW_SECONDS = config_int(
    "COSMOS3_LONG_WINDOW_SECONDS", minimum=30
)
COSMOS3_LONG_WINDOW_BURSTS = config_int(
    "COSMOS3_LONG_WINDOW_BURSTS", minimum=3, maximum=4
)
COSMOS3_LONG_WINDOW_SOURCE_FPS = config_float(
    "COSMOS3_LONG_WINDOW_SOURCE_FPS", minimum=0.25, maximum=2.0
)
COSMOS3_CUT_FPS = config_float(
    "COSMOS3_CUT_FPS", minimum=2.0, maximum=8.0
)
COSMOS3_VIDEO_WIDTH = config_int(
    "COSMOS3_VIDEO_WIDTH", minimum=224, maximum=720
)
COSMOS3_MIN_WALKING_FRACTION = config_float(
    "COSMOS3_MIN_WALKING_FRACTION", minimum=0.0, maximum=1.0
)
COSMOS3_MAX_PROMOTION_FRACTION = config_float(
    "COSMOS3_MAX_PROMOTION_FRACTION", minimum=0.0, maximum=1.0
)
if VISUAL_MODEL_BACKEND == "cosmos3":
    VISUAL_REVIEW_VERSION = (
        "cosmos3_nano_reasoner_drone_aerial_exclusion_v13"
        if COSMOS3_FAST_MODE
        else "cosmos3_nano_reasoner_hybrid_cuts_drone_aerial_cuda_v13"
    )
else:
    VISUAL_REVIEW_VERSION = (
        f"{VISUAL_MODEL_BACKEND}_scene_cuts_drone_aerial_v4"
    )

MIN_SEGMENT_DURATION_SECONDS = config_int(
    "MIN_SEGMENT_DURATION_SECONDS", minimum=1
)
MAX_SEGMENT_REVIEW_SECONDS = config_int(
    "MAX_SEGMENT_REVIEW_SECONDS", minimum=1
)
if MAX_SEGMENT_REVIEW_SECONDS < (2 * MIN_SEGMENT_DURATION_SECONDS) - 1:
    raise ValueError(
        "MAX_SEGMENT_REVIEW_SECONDS must be at least twice "
        "MIN_SEGMENT_DURATION_SECONDS minus one so long segments can be "
        "split without creating an undersized remainder."
    )
MIN_CUT_CONFIDENCE = config_float(
    "MIN_CUT_CONFIDENCE", minimum=0.0, maximum=1.0
)
MIN_SEGMENT_CONFIDENCE = config_float(
    "MIN_SEGMENT_CONFIDENCE", minimum=0.0, maximum=1.0
)
CUT_VERIFICATION_SECONDS = config_int(
    "CUT_VERIFICATION_SECONDS", minimum=2
)

PIPELINE_MODE = config_text("PIPELINE_MODE").lower()
if PIPELINE_MODE not in _VALID_PIPELINE_MODES:
    allowed_modes = ", ".join(sorted(_VALID_PIPELINE_MODES))
    raise ValueError(
        f"PIPELINE_MODE must be one of: {allowed_modes}. "
        f"Received: {PIPELINE_MODE!r}"
    )

SEQUENTIAL_DEVICE = config_device("SEQUENTIAL_DEVICE")
TEXT_DEVICE = config_device("TEXT_DEVICE")
VLM_DEVICE = config_device("VLM_DEVICE")
TEXT_MAX_OUTSTANDING = config_int("TEXT_MAX_OUTSTANDING", minimum=1)
VISUAL_MAX_OUTSTANDING = config_int(
    "VISUAL_MAX_OUTSTANDING", minimum=1
)
WORKER_POLL_SECONDS = config_float("WORKER_POLL_SECONDS", minimum=0.01)
WORKER_START_TIMEOUT_SECONDS = config_int(
    "WORKER_START_TIMEOUT_SECONDS", minimum=30
)

VIDEO_FORMAT = config_text("WALK_VIDEO_FORMAT")
YT_DLP_JS_RUNTIME = config_text("YT_DLP_JS_RUNTIME")
YT_DLP_REMOTE_COMPONENT = config_text("YT_DLP_REMOTE_COMPONENT")
YT_DLP_RETRIES = config_int("YT_DLP_RETRIES", minimum=0)
YT_DLP_FRAGMENT_RETRIES = config_int(
    "YT_DLP_FRAGMENT_RETRIES", minimum=0
)
YT_DLP_FILE_ACCESS_RETRIES = config_int(
    "YT_DLP_FILE_ACCESS_RETRIES", minimum=0
)
YT_DLP_DOWNLOAD_ATTEMPTS = config_int(
    "YT_DLP_DOWNLOAD_ATTEMPTS", minimum=1
)
YT_DLP_RETRY_SLEEP = config_text("YT_DLP_RETRY_SLEEP")
YT_DLP_FRAGMENT_RETRY_SLEEP = config_text(
    "YT_DLP_FRAGMENT_RETRY_SLEEP"
)
YT_DLP_HTTP_CHUNK_SIZE = config_text("YT_DLP_HTTP_CHUNK_SIZE")
YT_DLP_SOCKET_TIMEOUT_SECONDS = config_int(
    "YT_DLP_SOCKET_TIMEOUT_SECONDS", minimum=1
)
YT_DLP_DOWNLOAD_TIMEOUT_SECONDS = config_int(
    "YT_DLP_DOWNLOAD_TIMEOUT_SECONDS", minimum=1
)
_cookie_file = config_optional_text("YT_DLP_COOKIE_FILE")
if _cookie_file:
    _cookie_path = Path(_cookie_file).expanduser()
    if not _cookie_path.is_absolute():
        _cookie_path = Path(root_dir) / _cookie_path
    YT_DLP_COOKIE_FILE = str(_cookie_path)
else:
    YT_DLP_COOKIE_FILE = ""
YT_DLP_COOKIES_FROM_BROWSER = (
    config_optional_text("YT_DLP_COOKIES_FROM_BROWSER") or ""
)
if YT_DLP_COOKIE_FILE and YT_DLP_COOKIES_FROM_BROWSER:
    raise ValueError(
        "Set only one of YT_DLP_COOKIE_FILE or "
        "YT_DLP_COOKIES_FROM_BROWSER in config."
    )

SCENE_THRESHOLD = config_float(
    "SCENE_THRESHOLD", minimum=0.0, maximum=1.0
)
SCDET_THRESHOLD = config_float(
    "SCDET_THRESHOLD", minimum=0.0, maximum=100.0
)
CUT_DETECTION_BACKEND = config_text("CUT_DETECTION_BACKEND").lower()
if CUT_DETECTION_BACKEND not in _VALID_CUT_DETECTION_BACKENDS:
    allowed_cut_backends = ", ".join(
        sorted(_VALID_CUT_DETECTION_BACKENDS)
    )
    raise ValueError(
        "CUT_DETECTION_BACKEND must be one of: "
        f"{allowed_cut_backends}. Received: {CUT_DETECTION_BACKEND!r}"
    )
CUT_DETECTION_FPS = config_float(
    "CUT_DETECTION_FPS", minimum=1.0, maximum=30.0
)
CUT_DETECTION_WIDTH = config_int(
    "CUT_DETECTION_WIDTH", minimum=160, maximum=1920
)
CUT_DETECTION_CPU_FALLBACK = config_bool(
    "CUT_DETECTION_CPU_FALLBACK"
)
MERGE_NEARBY_SEC = config_float("MERGE_NEARBY_SEC", minimum=0.0)
IGNORE_FIRST_SEC = config_float("IGNORE_FIRST_SEC", minimum=0.0)
IGNORE_LAST_SEC = config_float("IGNORE_LAST_SEC", minimum=0.0)
CUT_DETECTION_TIMEOUT = config_int("CUT_DETECTION_TIMEOUT", minimum=1)

ENABLE_GEOCODING = config_bool("ENABLE_GEOCODING")
GEOCODER_USER_AGENT = config_text("GEOCODER_USER_AGENT")
GEOCODER_DELAY_SECONDS = config_float(
    "GEOCODER_DELAY_SECONDS", minimum=0.0
)

TIME_OF_DAY_CODES = {
    "day": 0,
    "night": 1,
    "dawn_dusk": 2,
    "unknown": -1,
}
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
    "walking_environment",
    "timestamp_labels",
    "embedded_location_text",
    "location_source",
    "start_time",
    "end_time",
    "upload_date",
    "channel",
]

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}