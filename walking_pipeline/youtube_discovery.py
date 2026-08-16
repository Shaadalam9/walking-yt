"""YouTube Data API discovery with API key rotation."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import settings
from .shared import clean_text, log, optional_text, save_state


YOUTUBE_DISCOVERY_SCHEMA_VERSION = "walking_continuous_batches_v2"
DISCOVERY_PROGRESS_KEY = "youtube_discovery_progress"


def load_api_keys() -> List[str]:
    raw: Any = ""
    try:
        import common  # type: ignore

        try:
            raw = common.get_secrets("google-api-keys")
        except KeyError:
            raw = common.get_secrets("google-api-key")
    except Exception:
        raw = ""

    if isinstance(raw, str):
        candidates: Iterable[Any] = re.split(r"[;,\n]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        candidates = raw
    else:
        candidates = [raw]

    result: List[str] = []
    for item in candidates:
        key = clean_text(item)
        if key and key not in result:
            result.append(key)
    return result


def split_queries(value: Optional[str]) -> List[str]:
    if not value:
        return list(settings.DEFAULT_WALKING_QUERIES)

    queries: List[str] = []
    for item in re.split(r"[\n;,]+", value):
        query = clean_text(item)
        if query and query not in queries:
            queries.append(query)
    return queries or list(settings.DEFAULT_WALKING_QUERIES)


def normalise_published_bound(value: Optional[str]) -> Optional[str]:
    text = optional_text(value)
    if not text or text.upper() == "YYYY-MM-DD":
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00Z"
    return text


def duration_to_seconds(iso_duration: str) -> int:
    match = re.fullmatch(
        r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        iso_duration or "",
    )
    if not match:
        return 0
    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def video_duration_is_eligible(duration: int) -> bool:
    """Return whether a video meets the configured minimum duration."""
    return duration >= settings.MIN_VIDEO_DURATION_SECONDS


class YouTubeDiscovery:
    def __init__(self, api_keys: List[str]) -> None:
        if not api_keys:
            raise RuntimeError(
                "No YouTube API key found. Add google-api-keys or "
                "google-api-key to the root secret file."
            )
        self.api_keys = api_keys
        self.key_index = 0
        self.client = self._build_client()

    def _build_client(self) -> Any:
        from googleapiclient.discovery import build  # type: ignore

        return build(
            "youtube",
            "v3",
            developerKey=self.api_keys[self.key_index],
            cache_discovery=False,
        )

    def _rotate_key(self) -> bool:
        self.key_index += 1
        if self.key_index >= len(self.api_keys):
            return False
        self.client = self._build_client()
        log(
            "Rotated to YouTube API key "
            f"{self.key_index + 1}/{len(self.api_keys)}"
        )
        return True

    def execute(self, request_factory: Callable[[Any], Any]) -> Dict[str, Any]:
        while True:
            try:
                response = request_factory(self.client).execute()
                return response if isinstance(response, dict) else {}
            except Exception as exc:
                message = str(exc)
                quota_error = any(
                    token in message
                    for token in (
                        "quotaExceeded",
                        "dailyLimitExceeded",
                        "rateLimitExceeded",
                    )
                )
                if quota_error and self._rotate_key():
                    continue
                raise

    def discover(self, state: Dict[str, Any]) -> int:
        queries = list(settings.DEFAULT_WALKING_QUERIES)
        published_after = normalise_published_bound(settings.PUBLISHED_AFTER)
        published_before = normalise_published_bound(settings.PUBLISHED_BEFORE)
        videos: Dict[str, Any] = state.setdefault("videos", {})
        progress = self._load_progress(state, queries)
        pending = progress["pending_candidates"]
        discovered_count = self._promote_pending_candidates(
            pending, videos
        )
        known_ids = set(videos)
        known_ids.update(
            str(item.get("video_id"))
            for item in pending
            if isinstance(item, dict) and item.get("video_id")
        )
        if self._limit_reached(discovered_count):
            save_state(state)
            return discovered_count

        page_tokens = progress["page_tokens"]
        exhausted_queries = set(progress["exhausted_queries"])
        query_count = len(queries)
        start_index = int(progress["next_query_index"]) % query_count

        for offset in range(query_count):
            query_index = (start_index + offset) % query_count
            query = queries[query_index]
            if query in exhausted_queries:
                continue

            page_token = optional_text(page_tokens.get(query))
            log(f"YouTube API search: {query}")

            for _ in range(settings.MAX_PAGES_PER_QUERY):
                if self._limit_reached(discovered_count):
                    progress["next_query_index"] = (
                        query_index + 1
                    ) % query_count
                    progress["exhausted_queries"] = sorted(
                        exhausted_queries
                    )
                    save_state(state)
                    return discovered_count

                def search_request(client: Any) -> Any:
                    parameters: Dict[str, Any] = {
                        "q": query,
                        "part": "snippet",
                        "type": "video",
                        "maxResults": settings.RESULTS_PER_PAGE,
                    }
                    if page_token:
                        parameters["pageToken"] = page_token
                    if published_after:
                        parameters["publishedAfter"] = published_after
                    if published_before:
                        parameters["publishedBefore"] = published_before
                    return client.search().list(**parameters)

                try:
                    search_response = self.execute(search_request)
                except Exception as exc:
                    log(f"YouTube search failed for query '{query}': {exc}")
                    break

                page_ids, search_snippets = self._new_page_ids(
                    search_response, known_ids
                )
                if page_ids:
                    candidate_records = self._load_video_details(
                        query=query,
                        page_ids=page_ids,
                        search_snippets=search_snippets,
                    )
                    if candidate_records is None:
                        break

                    remaining = self._remaining_limit(discovered_count)
                    if remaining is None:
                        accepted_records = candidate_records
                        deferred_records: List[tuple[str, Dict[str, Any]]] = []
                    else:
                        accepted_records = candidate_records[:remaining]
                        deferred_records = candidate_records[remaining:]

                    for video_id, record in accepted_records:
                        if video_id in videos:
                            continue
                        videos[video_id] = record
                        known_ids.add(video_id)
                        discovered_count += 1

                    for video_id, record in deferred_records:
                        if video_id in known_ids:
                            continue
                        pending.append(
                            {"video_id": video_id, "record": record}
                        )
                        known_ids.add(video_id)

                page_token = optional_text(
                    search_response.get("nextPageToken")
                )
                page_tokens[query] = page_token
                if not page_token:
                    exhausted_queries.add(query)
                    break

                progress["next_query_index"] = (
                    query_index + 1
                ) % query_count
                progress["exhausted_queries"] = sorted(
                    exhausted_queries
                )
                save_state(state)

                if self._limit_reached(discovered_count):
                    return discovered_count

            progress["next_query_index"] = (
                query_index + 1
            ) % query_count
            progress["exhausted_queries"] = sorted(exhausted_queries)
            save_state(state)

        if len(exhausted_queries) == query_count:
            progress["page_tokens"] = {}
            progress["exhausted_queries"] = []
            progress["next_query_index"] = 0
            progress["completed_sweeps"] = (
                int(progress.get("completed_sweeps", 0)) + 1
            )
            log(
                "Completed a full YouTube search sweep. The next empty "
                "batch check will start from the newest result pages."
            )
        save_state(state)

        return discovered_count

    @staticmethod
    def _load_progress(
        state: Dict[str, Any], queries: List[str]
    ) -> Dict[str, Any]:
        progress = state.get(DISCOVERY_PROGRESS_KEY)
        if (
            not isinstance(progress, dict)
            or progress.get("schema_version")
            != YOUTUBE_DISCOVERY_SCHEMA_VERSION
            or progress.get("queries") != queries
        ):
            progress = {
                "schema_version": YOUTUBE_DISCOVERY_SCHEMA_VERSION,
                "queries": list(queries),
                "next_query_index": 0,
                "page_tokens": {},
                "exhausted_queries": [],
                "pending_candidates": [],
                "completed_sweeps": 0,
            }
            state[DISCOVERY_PROGRESS_KEY] = progress

        if not isinstance(progress.get("page_tokens"), dict):
            progress["page_tokens"] = {}
        if not isinstance(progress.get("exhausted_queries"), list):
            progress["exhausted_queries"] = []
        if not isinstance(progress.get("pending_candidates"), list):
            progress["pending_candidates"] = []
        if not isinstance(progress.get("next_query_index"), int):
            progress["next_query_index"] = 0
        return progress

    def _promote_pending_candidates(
        self,
        pending: List[Any],
        videos: Dict[str, Any],
    ) -> int:
        promoted = 0
        while pending and not self._limit_reached(promoted):
            item = pending.pop(0)
            if not isinstance(item, dict):
                continue
            video_id = optional_text(item.get("video_id"))
            record = item.get("record")
            if (
                not video_id
                or video_id in videos
                or not isinstance(record, dict)
            ):
                continue
            videos[video_id] = record
            promoted += 1
        return promoted

    @staticmethod
    def _new_page_ids(
        response: Dict[str, Any], known_ids: set[str]
    ) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        page_ids: List[str] = []
        snippets: Dict[str, Dict[str, Any]] = {}
        for item in response.get("items", []):
            video_id = (
                item.get("id", {}).get("videoId")
                if isinstance(item, dict)
                else None
            )
            if not video_id or video_id in known_ids:
                continue
            page_ids.append(str(video_id))
            snippet = item.get("snippet", {})
            snippets[str(video_id)] = (
                snippet if isinstance(snippet, dict) else {}
            )
        return page_ids, snippets

    def _load_video_details(
        self,
        query: str,
        page_ids: List[str],
        search_snippets: Dict[str, Dict[str, Any]],
    ) -> Optional[List[tuple[str, Dict[str, Any]]]]:
        joined_ids = ",".join(page_ids)

        def details_request(client: Any) -> Any:
            return client.videos().list(
                part="snippet,contentDetails",
                id=joined_ids,
                maxResults=50,
            )

        try:
            response = self.execute(details_request)
        except Exception as exc:
            log(f"Could not fetch video details: {exc}")
            return None

        records: List[tuple[str, Dict[str, Any]]] = []
        for item in response.get("items", []):
            if not isinstance(item, dict):
                continue
            video_id = optional_text(item.get("id"))
            if not video_id:
                continue

            snippet = item.get("snippet", {})
            if not isinstance(snippet, dict):
                snippet = search_snippets.get(video_id, {})
            content = item.get("contentDetails", {})
            if not isinstance(content, dict):
                content = {}

            duration = duration_to_seconds(str(content.get("duration", "")))
            if not video_duration_is_eligible(duration):
                continue

            records.append(
                (
                    video_id,
                    self._new_video_record(
                        video_id, query, snippet, duration
                    ),
                )
            )
        return records

    @staticmethod
    def _new_video_record(
        video_id: str,
        query: str,
        snippet: Dict[str, Any],
        duration: int,
    ) -> Dict[str, Any]:
        return {
            "metadata": {
                "video_id": video_id,
                "title": optional_text(snippet.get("title")),
                "description": optional_text(snippet.get("description")),
                "channel": optional_text(snippet.get("channelId")),
                "channel_title": optional_text(snippet.get("channelTitle")),
                "upload_date": optional_text(snippet.get("publishedAt")),
                "duration_seconds": duration,
                "source_query": query,
                "video_url": (
                    f"https://www.youtube.com/watch?v={video_id}"
                ),
            },
            "status": "discovered",
            "text_decision": None,
            "visual_decision": None,
            "cuts": None,
            "segments": None,
            "error": None,
        }

    @staticmethod
    def _limit_reached(count: int) -> bool:
        return (
            settings.MAX_NEW_CANDIDATES is not None
            and count >= settings.MAX_NEW_CANDIDATES
        )

    @staticmethod
    def _remaining_limit(discovered_count: int) -> Optional[int]:
        if settings.MAX_NEW_CANDIDATES is None:
            return None
        return max(0, settings.MAX_NEW_CANDIDATES - discovered_count)