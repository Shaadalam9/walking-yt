# Walking pipeline v23 download guard

This update preserves all v22 behavior and prevents a failed YouTube video
from blocking the batch indefinitely.

## Behavior

- A permanently unavailable, private, removed, region-blocked, or members-only
  video is marked as a terminal skip immediately.
- Any other video-specific download failure is retried in the next cycle once.
  If the second attempt also fails, that video is marked as a terminal skip.
- Terminal download skips use `status: visual_rejected` for compatibility with
  the existing batch-completion logic. `download_terminal_reason` distinguishes
  `permanently_unavailable` from `download_retries_exhausted`.
- A cookie-file or YouTube authentication failure is treated as a system
  problem. The affected video remains pending and is never falsely rejected.
- File-based cookies are checked for existence, nonzero size, a Netscape
  header, and at least one cookie record before yt-dlp runs.
- Authentication failures stop prefetch and wait 60 seconds before another
  attempt. This protects YouTube and the account from a tight retry loop.
- The ordinary no-progress delay remains zero on SPIKE-1.
- The segment retry, saved-segment resume, persistent cookie path, explicit
  YouTube player clients, and retained Cosmos model from v22 are unchanged.
- `VISUAL_REVIEW_VERSION` is unchanged, so completed videos are not
  reanalysed merely because v23 is installed.

With the current stuck video, the first v23 cycle will log a permanent skip.
The following zero-delay cycle will see that the existing batch is complete
and resume YouTube API discovery.

## Install

Copy these files into the existing repository:

- `settings.py` -> `walking_pipeline/settings.py`
- `video_filter.py` -> `walking_pipeline/video_filter.py`
- `config.spike1` -> `config` for the SPIKE-1 image build
- `default.config` -> `default.config`

Do not replace the current persistent cookie file. v23 continues to read:

```text
/mnt/walking-yt/runtime-secrets/cookies.txt
```

## Verify locally

```bash
rm -rf walking_pipeline/__pycache__
python3 -m json.tool config >/dev/null
python3 -m py_compile walking_pipeline/settings.py walking_pipeline/video_filter.py
python3 -c '
from walking_pipeline import settings, video_filter
print(settings.SETTINGS_SCHEMA_VERSION)
print(settings.CONTINUOUS_IDLE_PAUSE_SECONDS)
print(settings.VIDEO_DOWNLOAD_FAILURES_BEFORE_SKIP)
print(settings.YT_DLP_AUTH_FAILURE_PAUSE_SECONDS)
print(settings.YT_DLP_EXTRACTOR_ARGS)
print(settings.YT_DLP_COOKIE_FILE)
print(video_filter.VIDEO_FILTER_SCHEMA_VERSION)
'
```

Expected SPIKE values:

```text
walking_download_retry_and_cookie_guard_v23
0
2
60
youtube:player_client=default,web_embedded
/mnt/walking-yt/runtime-secrets/cookies.txt
walking_download_retry_and_cookie_guard_v23
```

## Build the next image

Suggested tag:

```text
harbor.spike.tue.nl/globalwalk/walking-yt:spike-v8-download-guard
```

Build with the same Dockerfile and secrets used for v7. Stop or suspend v7
before starting v8 because both workloads share the same state file and PVC.

## Expected log messages

Permanent unavailable video:

```text
Skipping permanently unavailable video VIDEO_ID; continuing with the next candidate
```

Temporary failure, first attempt:

```text
Download failed for VIDEO_ID: ...; will retry in the next cycle (1 attempt(s) remaining)
```

Temporary failure, second attempt:

```text
Download still failed for VIDEO_ID after 2 attempt(s); skipping this video and continuing with the next candidate
```

Cookie or authentication problem:

```text
YouTube cookie/authentication failure while downloading VIDEO_ID; the video remains pending and will not be rejected
YouTube authentication is blocked. Verify or refresh the configured cookie file (...); waiting 60 second(s) before the next attempt.
```

Valid cookies:

```text
Using validated YouTube cookie file: /mnt/walking-yt/runtime-secrets/cookies.txt
```
