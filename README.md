# Walking YouTube Video Segmentation Pipeline

[![Python 3.12.13](https://img.shields.io/badge/python-3.12.13-blue.svg)](https://www.python.org/downloads/release/python-31213/)
[![Package manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

This project discovers long walking videos on YouTube, filters unsuitable videos, downloads accepted candidates, detects scene boundaries, identifies genuine first-person walking segments, extracts explicit location evidence, and writes resumable research outputs.

The code is open source and intended primarily for academic research. Forks, pull requests, and contributions in the spirit of open science are welcome. For collaboration inquiries, contact Md Shadab Alam at `md_shadab_alam@outlook.com` or Pavlo Bazilinskyy at `pavlo.bazilinskyy@gmail.com`.

## Contents

- [Pipeline overview](#pipeline-overview)
- [Selection and location policy](#selection-and-location-policy)
- [Requirements and installation](#requirements-and-installation)
- [Credentials and cookies](#credentials-and-cookies)
- [Running locally](#running-locally)
- [Configuration reference](#configuration-reference)
- [Continuous operation and resuming](#continuous-operation-and-resuming)
- [Outputs](#outputs)
- [Running on SPIKE 1](#running-on-spike-1)
- [Performance and monitoring](#performance-and-monitoring)
- [Troubleshooting](#troubleshooting)
- [Security](#security)

## Pipeline overview

1. **YouTube discovery** searches the configured walking-related queries and records new candidates.
2. **Text filtering** uses Qwen to reject clearly unsuitable metadata before download.
3. **Atomic downloading** writes into `walking_videos/.tmp`, validates the file, and then moves it into `walking_videos`.
4. **Queue prefetching** keeps videos ready while another video is analysed.
5. **Intro detection** finds where the main walking content starts.
6. **CUDA cut detection** uses FFmpeg hardware decoding and GPU scaling, with an optional CPU fallback.
7. **Visual review** uses NVIDIA Cosmos 3 Nano.
8. **Hard exclusions** remove drone/aerial footage and other non-walking content.
9. **Location extraction** uses only author timestamps/chapters and readable embedded text.
10. **Incremental output** updates JSON state and the locality CSV as work finishes.

Current versions:

```text
Settings:      walking_single_gpu_throughput_v20
Video filter:  walking_single_gpu_throughput_v20
Visual review: cosmos3_nano_reasoner_hybrid_cuts_drone_aerial_cuda_v13
```

## Selection and location policy

Accepted content shows genuine walking movement from a pedestrian or first-person perspective. Street walks, parks, markets, squares, waterfronts, and indoor public areas may be accepted.

Rejected content includes:

- drone or other aerial footage;
- static shots and slideshows;
- driving, cycling, boating, or other vehicle-only movement;
- promotional montages, maps, title cards, and introductions;
- segments with insufficient walking content;
- segments below the configured duration or confidence thresholds.

The pipeline intentionally does **not** infer a location from general visual appearance. It records a location only when supported by an author-provided timestamp/chapter label, readable location text embedded in the video, or both. Otherwise, `location_source` is `none`.

## Repository layout

```text
walking-yt/
├── main.py                         Pipeline entry point
├── config                          Active JSON configuration
├── default.config                  Configuration template
├── secret                          YouTube API keys
├── cookies.txt                     Netscape-format YouTube cookies
├── common.py                       Configuration and secret loading
├── Dockerfile.spike1               SPIKE 1 image definition
├── pyproject.toml                  Python project metadata
├── uv.lock                         Locked dependencies
└── walking_pipeline/
    ├── settings.py                 Typed configuration
    ├── runtime.py                  Runtime validation
    ├── pipeline.py                 Continuous orchestration
    ├── youtube_discovery.py        YouTube discovery
    ├── text_filter.py              Metadata filtering
    ├── video_filter.py             Download and visual analysis
    ├── cut_detection.py            CUDA/CPU cut detection
    ├── location.py                 Location resolution
    └── output_writer.py            JSON and CSV writing
```

## Requirements and installation

Required: Python 3.12.13, `uv`, Git, FFmpeg/`ffprobe`, Deno, the locked yt-dlp installation, a production CUDA GPU, API credentials, and valid YouTube cookies.

Install `uv` on macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Create and activate the environment:

```bash
uv sync --frozen
source .venv/bin/activate
```

Windows:

```powershell
uv sync --frozen
.\.venv\Scripts\Activate.ps1
```

Verify tools:

```bash
ffmpeg -version
ffprobe -version
deno --version
yt-dlp --version
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -decoders | grep cuvid
ffmpeg -hide_banner -filters | grep scale_cuda
```

On Windows, replace `grep` with `Select-String`.

## Credentials and cookies

Keep `config`, `secret`, and `cookies.txt` in the repository root. `secret` is loaded through `common.py` and contains YouTube Data API keys; preserve `default.secret` structure.

Export `cookies.txt` in Netscape format. Its first line should be:

```text
# Netscape HTTP Cookie File
```

Test it:

```bash
yt-dlp \
  --cookies cookies.txt \
  --js-runtimes deno \
  --remote-components "ejs:github" \
  --skip-download \
  --print "%(id)s | %(title)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

Export fresh cookies when YouTube reports `Sign in to confirm you’re not a bot`. Ignore credentials:

```gitignore
/cookies.txt
secret
```

If previously tracked:

```bash
git rm --cached cookies.txt secret
```

## Running locally

Validate:

```bash
python3 -m json.tool config >/dev/null
python3 -m py_compile \
  walking_pipeline/settings.py \
  walking_pipeline/video_filter.py \
  walking_pipeline/cut_detection.py
```

Windows PowerShell:

```powershell
python -m json.tool .\config > $null
python -m py_compile `
  .\walking_pipeline\settings.py `
  .\walking_pipeline\video_filter.py `
  .\walking_pipeline\cut_detection.py
```

Do not use SPIKE paths locally. At minimum:

```json
{
  "SPIKE1_MODE": false,
  "SPIKE1_REQUIRE_GPU": false,
  "SPIKE1_REQUIRE_PERSISTENT_STORAGE": false,
  "WALK_DATA_DIR": "data",
  "HF_HOME": null,
  "HF_HUB_CACHE": null,
  "YT_DLP_COOKIE_FILE": "cookies.txt",
  "COSMOS3_BATCH_SIZE": 1
}
```

For one GPU, use `PIPELINE_MODE: sequential` and `cuda:0` for all device values. Start with `python3 main.py` on macOS/Linux or `python main.py` on Windows. Stop with `Ctrl+C`; the next run resumes from JSON state.

## Configuration reference

The active `config` file is JSON. Booleans must be `true` or `false`; disabled optional values should normally be `null`. Relative walking-pipeline paths resolve below `WALK_DATA_DIR`.

The tables document every key in the current 102-parameter configuration.

### Compatibility paths

These support older or adjacent repository components and do not normally control the walking pipeline.

| Parameter | Type / values | Meaning |
|---|---|---|
| `data` | Path string | Base directory for earlier bounding-box/dataset code. |
| `mapping` | File path | Mapping CSV used by earlier dataset code. |
| `output_dir` | Directory path | Output directory for earlier non-walking components. |
| `VIDEO_BASE_URL` | HTTP/HTTPS URL | Base URL used when earlier components construct hosted video URLs. |
| `videos` | Directory path | Earlier video directory; separate from `WALK_VIDEO_DIR`. |

### SPIKE, storage, and libraries

| Parameter | Type / values | Meaning |
|---|---|---|
| `SPIKE1_MODE` | Boolean | Enable SPIKE validation. Use `true` on SPIKE and `false` locally. |
| `SPIKE1_REQUIRE_GPU` | Boolean | Fail startup if CUDA is unavailable. |
| `SPIKE1_REQUIRE_PERSISTENT_STORAGE` | Boolean | Require persistent rather than container-local data. |
| `WALK_DATA_DIR` | Directory path | Root of state, CSV, videos, and cache; `/mnt/walking-yt` on SPIKE or `data` locally. |
| `WALK_STATE_JSON` | File name/path | Authoritative resumable state. |
| `WALK_OUTPUT_CSV` | File name/path | Incremental locality CSV. |
| `WALK_VIDEO_DIR` | Directory name/path | Validated downloads ready for analysis. |
| `WALK_VIDEO_TMP_DIR` | Directory name/path | Atomic temporary downloads; reset at startup and best kept on the same filesystem as `WALK_VIDEO_DIR`. |
| `WALK_GEOCODE_CACHE` | File name/path | Persistent geocoding cache. |
| `HF_HOME` | Path or `null` | Hugging Face home; use persistent storage on SPIKE. |
| `HF_HUB_CACHE` | Path or `null` | Hugging Face Hub cache. |
| `HF_XET_HIGH_PERFORMANCE` | Boolean | Enable high-performance Xet transfers when supported. |
| `TOKENIZERS_PARALLELISM` | Boolean | Control tokenizer parallelism; `false` avoids excessive threads. |
| `PYTORCH_CUDA_ALLOC_CONF` | String or `null` | Optional allocator settings such as `expandable_segments:True`. |

### Discovery and continuous batches

| Parameter | Type / values | Meaning |
|---|---|---|
| `WALKING_QUERIES` | Array of strings | YouTube search phrases. Broader terms increase candidates and false positives. |
| `MAX_PAGES_PER_QUERY` | Positive integer | Search pages read per query and discovery cycle. |
| `RESULTS_PER_PAGE` | Integer 1–50 | YouTube results per page. |
| `MAX_NEW_CANDIDATES` | Positive integer | Maximum unseen candidates added per discovery; `200` gives 200-candidate batches. |
| `MAX_VIDEOS_PER_RUN` | Positive integer | Maximum videos handled per cycle. |
| `CONTINUOUS_BATCH_MODE` | Boolean | `true` repeats cycles; `false` exits after one. |
| `CONTINUOUS_BATCH_PAUSE_SECONDS` | Number ≥ 0 | Delay between active cycles. |
| `CONTINUOUS_IDLE_PAUSE_SECONDS` | Number ≥ 0 | Delay before rediscovery after an idle cycle. |
| `MIN_VIDEO_DURATION_SECONDS` | Integer ≥ 0 | Reject shorter videos; `300` means five minutes. |
| `PUBLISHED_AFTER` | ISO date/datetime or `null` | Optional lower publication boundary. |
| `PUBLISHED_BEFORE` | ISO date/datetime or `null` | Optional upper publication boundary. |

### Download queue and retention

| Parameter | Type / values | Meaning |
|---|---|---|
| `VIDEO_DOWNLOAD_QUEUE_SIZE` | Integer ≥ 1 | Downloaded videos waiting in addition to the active video. `5` can mean five waiting plus one active. |
| `VIDEO_DOWNLOAD_FAILURE_ALLOWANCE_PER_RUN` | Integer ≥ 0 | Extra attempts while filling the queue after failures. |
| `VIDEO_DOWNLOAD_MAX_CONSECUTIVE_FAILURES` | Integer ≥ 1 | Stop prefetch after repeated failures. |
| `VIDEO_DOWNLOAD_STOP_ON_AUTH_ERROR` | Boolean | Stop new downloads on a recognised authentication block. |
| `DELETE_VIDEO_AFTER_PROCESSING` | Boolean | Delete MP4 only after `complete` or `visual_rejected` when `true`; retain it when `false`. |

### Text model

| Parameter | Type / values | Meaning |
|---|---|---|
| `TEXT_LLM_MODEL` | Model ID/path | Model used to filter metadata. |
| `TEXT_LLM_4BIT` | Boolean | Enable compatible 4-bit loading to save VRAM. |
| `TEXT_MAX_NEW_TOKENS` | Positive integer | Maximum generated tokens per text decision. |
| `MIN_TEXT_CONFIDENCE` | Number 0–1 | Acceptance threshold; higher is stricter. |

### Visual backends and Cosmos 3

| Parameter | Type / values | Meaning |
|---|---|---|
| `VISUAL_MODEL_BACKEND` | Implemented backend name | Select visual implementation. Production uses `cosmos3`. |
| `INTERNVL_MODEL` | Model ID/path | Legacy/alternative InternVL model; unused with the Cosmos backend. |
| `INTERNVL_4BIT` | Boolean | Enable compatible 4-bit InternVL loading. |
| `FRAMES_PER_SAMPLE` | Positive integer | Frames used by non-Cosmos visual sampling. |
| `VLM_MAX_NEW_TOKENS` | Positive integer | Generation limit for the alternative visual backend. |
| `COSMOS3_MODEL` | Model ID/path | Cosmos model; production uses `nvidia/Cosmos3-Nano`. |
| `COSMOS3_4BIT` | Boolean | Request compatible 4-bit Cosmos loading. Keep `false` unless tested. |
| `COSMOS3_MAX_NEW_TOKENS` | Positive integer | Maximum tokens per Cosmos response. |
| `COSMOS3_BATCH_SIZE` | Positive integer | Compatible reviews processed together. Use `4` on SPIKE and start with `1` locally. Reduce after out-of-memory errors. |
| `CLIP_PREPARE_WORKERS` | Positive integer | CPU workers preparing visual clips; `2` suits the eight-core SPIKE allocation. |
| `COSMOS3_FAST_MODE` | Boolean | Enable reduced-cost review. `false` retains the fuller review path. |
| `COSMOS3_INTRO_SEARCH_SECONDS` | Positive number | Initial duration searched for the main walking-content start. |
| `COSMOS3_INTRO_FPS` | Positive number | Intro-detection sampling rate. |
| `COSMOS3_MIN_CONTENT_START_CONFIDENCE` | Number 0–1 | Confidence required to move the start past an introduction. |
| `COSMOS3_FRAMES_PER_SAMPLE` | Positive integer | Frames in a standard Cosmos segment sample. |
| `COSMOS3_REVIEW_FPS` | Positive number | Sampling rate for standard segment review. |
| `COSMOS3_LONG_WINDOW_SECONDS` | Positive number | Duration represented by a long-window review. |
| `COSMOS3_LONG_WINDOW_BURSTS` | Positive integer | Temporal bursts sampled in a long window. |
| `COSMOS3_LONG_WINDOW_SOURCE_FPS` | Positive number | Source rate used to build long-window bursts. |
| `COSMOS3_CUT_FPS` | Positive number | Visual sampling around a cut; distinct from `CUT_DETECTION_FPS`. |
| `COSMOS3_VIDEO_WIDTH` | Positive even integer | Width of Cosmos review clips. Lower is faster but removes detail. |
| `COSMOS3_MIN_WALKING_FRACTION` | Number 0–1 | Minimum fraction of a segment estimated as walking. |
| `COSMOS3_MAX_PROMOTION_FRACTION` | Number 0–1 | Maximum promotional/non-content fraction. |

### Segment thresholds

| Parameter | Type / values | Meaning |
|---|---|---|
| `MIN_SEGMENT_DURATION_SECONDS` | Positive number | Reject shorter candidate segments. |
| `MAX_SEGMENT_REVIEW_SECONDS` | Positive number | Maximum duration represented in one review request. |
| `MIN_CUT_CONFIDENCE` | Number 0–1 | Confidence required to accept a verified cut. |
| `MIN_SEGMENT_CONFIDENCE` | Number 0–1 | Confidence required to keep a walking segment. |
| `CUT_VERIFICATION_SECONDS` | Positive number | Time sampled on each side of a boundary. |

### Device placement and workers

| Parameter | Type / values | Meaning |
|---|---|---|
| `PIPELINE_MODE` | `sequential`, `parallel`, or `auto` | Use `sequential` for one GPU. `auto` can choose parallel only when configured devices are visible. |
| `SEQUENTIAL_DEVICE` | Torch device | Device for both models in sequential mode, normally `cuda:0`. |
| `TEXT_DEVICE` | Torch device | Text-model device in parallel mode. |
| `VLM_DEVICE` | Torch device | Visual-model device in parallel mode; `cuda:0` for a one-GPU job. |
| `TEXT_MAX_OUTSTANDING` | Positive integer | Maximum queued text tasks in worker/parallel execution. |
| `VISUAL_MAX_OUTSTANDING` | Positive integer | Maximum queued visual tasks in worker/parallel execution. |
| `WORKER_POLL_SECONDS` | Positive number | Worker-message polling interval. |
| `WORKER_START_TIMEOUT_SECONDS` | Positive number | Maximum model-worker initialisation wait. |

### yt-dlp format and authentication

| Parameter | Type / values | Meaning |
|---|---|---|
| `WALK_VIDEO_FORMAT` | yt-dlp format expression | Stream selection. Current expression prefers 720p, then 480p, 360p, and 144p fallbacks. |
| `YT_DLP_COOKIE_FILE` | Path or `null` | Netscape cookie file; `cookies.txt` locally or `/run/secrets/walking-yt/cookies.txt` in the bundled image. |
| `YT_DLP_COOKIES_FROM_BROWSER` | Browser spec or `null` | Optional local browser-cookie source; keep `null` when using a file or inside SPIKE. |
| `YT_DLP_JS_RUNTIME` | Runtime spec | JavaScript runtime; use `deno`. |
| `YT_DLP_REMOTE_COMPONENT` | Component spec or `null` | Challenge component; current value `ejs:github` requires network access. |

### yt-dlp resilience

| Parameter | Type / values | Meaning |
|---|---|---|
| `YT_DLP_RETRIES` | Integer ≥ 0 | General yt-dlp retries. |
| `YT_DLP_FRAGMENT_RETRIES` | Integer ≥ 0 | Media-fragment retries. |
| `YT_DLP_FILE_ACCESS_RETRIES` | Integer ≥ 0 | Retries for locked/unavailable output files. |
| `YT_DLP_DOWNLOAD_ATTEMPTS` | Integer ≥ 1 | Whole yt-dlp command attempts by the pipeline. |
| `YT_DLP_RETRY_SLEEP` | yt-dlp expression | General retry delay, for example `exp=1:5`. |
| `YT_DLP_FRAGMENT_RETRY_SLEEP` | yt-dlp expression | Fragment retry delay. |
| `YT_DLP_HTTP_CHUNK_SIZE` | Size string or `null` | HTTP chunk size, for example `10M`. |
| `YT_DLP_SOCKET_TIMEOUT_SECONDS` | Positive number | Network socket timeout. |
| `YT_DLP_DOWNLOAD_TIMEOUT_SECONDS` | Positive number | Pipeline timeout for one complete download. |

### Cut detection

| Parameter | Type / values | Meaning |
|---|---|---|
| `SCENE_THRESHOLD` | Number 0–1 | FFmpeg scene threshold. Lower produces more candidates; higher produces fewer. |
| `SCDET_THRESHOLD` | Positive number | Threshold for the alternative detector. |
| `CUT_DETECTION_BACKEND` | Implemented backend | Use `ffmpeg_cuda` when CUDA, CUVID, and `scale_cuda` are available. |
| `CUT_DETECTION_FPS` | Positive number | Frames analysed per second. Higher improves temporal sensitivity but costs more. Production uses `6.0`. |
| `CUT_DETECTION_WIDTH` | Positive even integer | Scaled detection width. Production uses `320`. |
| `CUT_DETECTION_CPU_FALLBACK` | Boolean | Retry on CPU after CUDA failure. More resilient but much slower. |
| `MERGE_NEARBY_SEC` | Number ≥ 0 | Merge candidates closer than this interval. |
| `IGNORE_FIRST_SEC` | Number ≥ 0 | Ignore cuts near the beginning. |
| `IGNORE_LAST_SEC` | Number ≥ 0 | Ignore cuts near the end. |
| `CUT_DETECTION_TIMEOUT` | Positive number | Maximum cut subprocess runtime. |

### Geocoding

| Parameter | Type / values | Meaning |
|---|---|---|
| `ENABLE_GEOCODING` | Boolean | Resolve extracted place names to administrative fields and coordinates. |
| `GEOCODER_USER_AGENT` | Non-empty string | Application identifier sent to the geocoder. |
| `GEOCODER_DELAY_SECONDS` | Number ≥ 0 | Minimum delay between geocoder requests; `1.1` is conservative. |

### Recommended one-GPU SPIKE subset

```json
{
  "SPIKE1_MODE": true,
  "SPIKE1_REQUIRE_GPU": true,
  "SPIKE1_REQUIRE_PERSISTENT_STORAGE": true,
  "WALK_DATA_DIR": "/mnt/walking-yt",
  "HF_HOME": "/mnt/walking-yt/huggingface",
  "HF_HUB_CACHE": "/mnt/walking-yt/huggingface/hub",
  "MAX_NEW_CANDIDATES": 200,
  "MAX_VIDEOS_PER_RUN": 200,
  "CONTINUOUS_BATCH_MODE": true,
  "VIDEO_DOWNLOAD_QUEUE_SIZE": 5,
  "DELETE_VIDEO_AFTER_PROCESSING": true,
  "VISUAL_MODEL_BACKEND": "cosmos3",
  "COSMOS3_BATCH_SIZE": 4,
  "CLIP_PREPARE_WORKERS": 2,
  "PIPELINE_MODE": "sequential",
  "SEQUENTIAL_DEVICE": "cuda:0",
  "TEXT_DEVICE": "cuda:0",
  "VLM_DEVICE": "cuda:0",
  "CUT_DETECTION_BACKEND": "ffmpeg_cuda",
  "CUT_DETECTION_FPS": 6.0,
  "CUT_DETECTION_WIDTH": 320,
  "CUT_DETECTION_CPU_FALLBACK": true,
  "YT_DLP_COOKIE_FILE": "/run/secrets/walking-yt/cookies.txt"
}
```

## Continuous operation and resuming

With:

```json
{
  "MAX_NEW_CANDIDATES": 200,
  "MAX_VIDEOS_PER_RUN": 200,
  "CONTINUOUS_BATCH_MODE": true
}
```

the pipeline completes the existing discovery batch first. When no unfinished videos remain, it discovers up to 200 new candidates and starts the next cycle. It does not repeatedly call the YouTube search API while the current batch still contains unfinished candidates.

`walking_pipeline_state.json` is authoritative. Completed videos are skipped when their status and visual-review version are current. The CSV is a derived locality output and is not the sole deduplication source.

Changing `VISUAL_REVIEW_VERSION` intentionally makes earlier visual results stale, which can cause review again. Do not change schema/version constants merely to rename a release.

`VIDEO_DOWNLOAD_QUEUE_SIZE` counts waiting downloads, not the active video. A queue target of five can therefore result in six validated files: one active plus five waiting.

The startup cleanup removes only the temporary download directory. State, CSV, geocode cache, model cache, and completed downloads on the PVC survive restarts and Run:ai preemption.

## Outputs

With `WALK_DATA_DIR=/mnt/walking-yt`:

```text
/mnt/walking-yt/walking_pipeline_state.json
/mnt/walking-yt/walking_segments.csv
/mnt/walking-yt/walking_geocode_cache.json
/mnt/walking-yt/walking_videos/
/mnt/walking-yt/huggingface/
```

### JSON state

`walking_pipeline_state.json` stores candidates, metadata decisions, download status, visual review, segment boundaries, location evidence, versions, and errors. Use it to audit and resume the pipeline.

### CSV schema

`walking_segments.csv` is incrementally rebuilt from accepted state.

| Column | Meaning |
|---|---|
| `id` | Stable locality row identifier. |
| `locality` | Primary resolved locality. |
| `locality_aka` | Alternative locality names. |
| `state` | State/province or equivalent. |
| `country` | Country name. |
| `iso3` | ISO 3166-1 alpha-3 code. |
| `continent` | Continent. |
| `lat` / `lon` | Coordinates. |
| `videos` | Source video IDs. |
| `time_of_day` | Per-segment time labels. |
| `walking_environment` | Environment classes such as `street` or `waterfront`. |
| `timestamp_labels` | Author chapter/timestamp evidence. |
| `embedded_location_text` | OCR-derived location text. |
| `location_source` | `timestamp_description`, `embedded_video`, `both`, or `none`. |
| `start_time` / `end_time` | Segment boundaries in seconds. |
| `upload_date` | Source upload dates. |
| `channel` | Source YouTube channel IDs. |

List fields use compact bracket notation, for example `[2BfHMnDOteA,KRUtzHibboI]`. Vehicle type is deliberately omitted because the pipeline targets pedestrians.

## Running on SPIKE 1

### 1. Prepare and validate

Use `/mnt/walking-yt` for persistent data and `/run/secrets/walking-yt/cookies.txt` for the bundled cookie path. Then:

```bash
python3 -m json.tool config >/dev/null
```

### 2. Build and push

The current workflow builds AMD64 on macOS and injects the three runtime files from BuildKit secret mounts:

```bash
IMAGE="harbor.spike.tue.nl/globalwalk/walking-yt:spike-v5-bundled"

docker login harbor.spike.tue.nl --username "YOUR_TUE_EMAIL"

docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.spike1 \
  --secret id=walking_config,src=config \
  --secret id=walking_secret,src=secret \
  --secret id=youtube_cookies,src=cookies.txt \
  --tag "$IMAGE" \
  --push \
  .
```

Use `--no-cache` only when cached layers must be rebuilt. Verify:

```bash
docker buildx imagetools inspect "$IMAGE"
```

The output must contain `Platform: linux/amd64`.

### 3. Create the Run:ai workload

| Setting | Value |
|---|---|
| Cluster | `spike-1` |
| Project | `globalwalk` |
| Type | Standard training |
| Image | `harbor.spike.tue.nl/globalwalk/walking-yt:spike-v5-bundled` |
| Pull policy | If not present |
| Registry secret | `dockerregistry-harbor-globalwalk` |
| GPU | 1 device, fractioning off |
| CPU | 8 cores / 8000 millicores |
| RAM | 64 GB request and limit |
| Data source | PVC `globalwalk` |
| Mount path | `/mnt/walking-yt` |
| Working directory | `/app` |
| Command/arguments | Empty; use the image command |
| Preemptibility | Preemptible |
| Priority | Low |
| Node pool | `default` |

Low-priority preemptible execution is the requested **over-quota** mode. It can wait for spare capacity or be interrupted; persistent state makes restart safe.

### 4. Security identity

Under **Security**, select **Custom**:

```text
UID: 20234844
GID: 20234844
Supplementary groups: empty
```

This avoids Kubernetes failing with `image has non-numeric user` while `runAsNonRoot` is enforced.

### 5. Start and inspect

```bash
runai login
runai cluster set spike-1
runai project set globalwalk
runai training standard describe walking-yt-v5 --project globalwalk
runai training standard logs walking-yt-v5 --project globalwalk --follow
```

## Performance and monitoring

The main historical bottleneck was CPU scene detection over multi-hour 60 FPS videos. The current CUDA path downsizes on the GPU and samples only the configured detection rate. Expected log:

```text
Running CUDA cut detection at 6 FPS and width 320
```

Safe tuning order:

1. Keep `CUT_DETECTION_BACKEND=ffmpeg_cuda`.
2. Use `CUT_DETECTION_FPS=6.0` and `CUT_DETECTION_WIDTH=320` as the baseline.
3. Use `COSMOS3_BATCH_SIZE=4` on the 192 GB SPIKE GPU.
4. Use `CLIP_PREPARE_WORKERS=2` with eight CPU cores.
5. Reduce Cosmos batch size to 2 or 1 after out-of-memory errors.
6. Raise detection FPS only if validation shows missed cuts.
7. Investigate any CPU fallback because it can be much slower.

Inspect activity:

```bash
runai training standard exec walking-yt-v5 \
  --project globalwalk \
  -- bash -lc 'date; pgrep -af ffmpeg || true; ps -eo pid,etime,%cpu,%mem,cmd --sort=-%cpu | head -n 15; nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,memory.used,memory.total --format=csv,noheader'
```

Low general GPU utilisation during cut detection can coexist with useful decoder utilisation because NVIDIA's decode engine is separate.

### Inspect persistent files

```bash
runai training standard exec walking-yt-v5 \
  --project globalwalk \
  -- bash -lc 'ls -lah /mnt/walking-yt; ls -lah /mnt/walking-yt/walking_videos | head'
```

### Download current results

```bash
mkdir -p walking-results && \
runai training standard exec walking-yt-v5 --project globalwalk -- cat /mnt/walking-yt/walking_segments.csv > walking-results/walking_segments.csv && \
runai training standard exec walking-yt-v5 --project globalwalk -- cat /mnt/walking-yt/walking_pipeline_state.json > walking-results/walking_pipeline_state.json
```

They are saved on the computer running the command under `~/walking-results/`. Validate:

```bash
python3 -m json.tool walking-results/walking_pipeline_state.json >/dev/null
head -n 3 walking-results/walking_segments.csv
```

The CSV may contain only its header while the first video is still under analysis. Downloads alone do not create segment rows.

## Troubleshooting

### Missing FFmpeg, Deno, or configuration values

Install the missing command and ensure it is on `PATH`. Every key read by `settings.py` must exist in `config`, even if its value is `null`. Validate with:

```bash
python3 -m json.tool config >/dev/null
```

### YouTube bot/authentication error

Export fresh Netscape cookies, verify `YT_DLP_COOKIE_FILE`, and run the standalone yt-dlp test. `VIDEO_DOWNLOAD_STOP_ON_AUTH_ERROR=true` prevents an endless failure loop.

### Partial download/read error

The pipeline writes into `.tmp` and promotes only validated files. Check connectivity and cookies, then restart. yt-dlp and pipeline retry settings handle transient errors.

### A completed video runs again

Check that:

- the same `WALK_DATA_DIR` is mounted;
- `walking_pipeline_state.json` still exists;
- the video reached a final state;
- `VISUAL_REVIEW_VERSION` did not change;
- the previous run was not stopped before the final state write.

The CSV alone does not mark a candidate complete.

### Cut detection is slow

Look for `Running CUDA cut detection`. If logs show CPU fallback, check CUDA, CUVID, and `scale_cuda` support. CPU fallback is resilient but can be dramatically slower.

### CUDA out of memory

Reduce `COSMOS3_BATCH_SIZE` from 4 to 2 or 1. Restarting is safe when state is on the PVC.

### Run:ai stays Initializing

Run:

```bash
runai training standard describe WORKLOAD_NAME --project globalwalk
```

Read pod events. If they mention a non-numeric image user, recreate with Custom UID/GID `20234844`.

### Run:ai stays Pending

This is normal for low-priority over-quota work when no spare GPU is available. Confirm the job is low priority, preemptible, and requests one GPU.

### CSV looks unusual

The file is valid CSV with nested research fields encoded as compact bracket strings. Use a CSV-aware reader. The JSON state retains the authoritative structured representation.

## Security

- Never commit `secret` or `cookies.txt`.
- Treat YouTube cookies as account credentials.
- The current bundled workflow copies `config`, `secret`, and `cookies.txt` into the final private image. Anyone who can pull or inspect it may recover them.
- Keep Harbor private, delete obsolete bundled tags when appropriate, and rotate credentials after possible exposure.
- Prefer runtime-mounted Run:ai/Kubernetes secrets for long-term production.
- Deleting a Run:ai workload does not delete persistent state or CSV files on the PVC.
- `DELETE_VIDEO_AFTER_PROCESSING=true` saves storage by removing an MP4 after final analysis; use `false` when source files must remain available for manual verification.

## Contributing

When adding a setting, update both `config` and `default.config` and document it here. Bump schema/review versions only when stored-result semantics change. Compile modified modules, test one local video and one SPIKE restart, and never commit credentials, videos, caches, or generated research data.
