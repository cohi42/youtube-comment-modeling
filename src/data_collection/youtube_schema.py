from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def iter_flat_comment_records(collection: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield one flat record per comment from a nested Shorts collection JSON."""

    for channel in collection.get("channels", []):
        channel_title = channel.get("channelTitle")
        channel_handle = channel.get("channelHandle")
        channel_id = channel.get("channelId")

        for video in channel.get("videos", []):
            stats = video.get("details", {}).get("statistics", {})
            snippet = video.get("details", {}).get("snippet", {})
            view_count = video.get("viewCount", stats.get("viewCount"))
            like_count = video.get("likeCount", stats.get("likeCount"))
            published_at = video.get("publishedAt", snippet.get("publishedAt"))

            for comment in video.get("comments", []):
                yield {
                    "channel_id": channel_id,
                    "channel_title": channel_title or snippet.get("channelTitle"),
                    "channel_handle": channel_handle,
                    "video_id": video.get("videoId"),
                    "video_url": video.get("url"),
                    "video_title": video.get("title", snippet.get("title")),
                    "video_published_at": published_at,
                    "video_view_count": safe_int(view_count),
                    "video_like_count": safe_int(like_count),
                    "comment_id": comment.get("commentId"),
                    "comment_author": comment.get("author"),
                    "comment_text": comment.get("text", ""),
                    "comment_like_count": safe_int(comment.get("likeCount", comment.get("like_count"))),
                    "comment_published_at": comment.get("publishedAt", comment.get("published_at")),
                    "comment_updated_at": comment.get("updatedAt"),
                }


def flatten_collection(collection: dict[str, Any]) -> list[dict[str, Any]]:
    return list(iter_flat_comment_records(collection))


def anonymize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that are not needed for public examples."""

    public_keys = {
        "video_id",
        "video_title",
        "video_published_at",
        "video_view_count",
        "comment_text",
        "comment_like_count",
        "comment_published_at",
        "hours_after_publish",
        "relevance_score",
        "preference_proxy_score",
    }
    return {key: value for key, value in record.items() if key in public_keys}

