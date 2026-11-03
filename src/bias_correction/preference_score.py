"""Final preference proxy score (double correction).

Original: create_candidate_dataset() in the
"댓글_시간_노출_편향_보정과 SFT" notebook.

    final_expected = ExpectedLikesByViews(views) * TimeAdvantage(timing)
    final_score    = likes / (final_expected + 1e-5)

The relevance score from the LLM judge is NOT multiplied in. In the project it
was used afterwards as the first sort key (see relevance_scoring/qwen_judge.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .time_decay import TimeDecayParams, hours_after_publish, time_advantage
from .view_scale import ViewScaleParams, expected_likes_by_views


@dataclass(frozen=True)
class PreferenceScoreConfig:
    time_params: TimeDecayParams = field(default_factory=TimeDecayParams)
    view_params: ViewScaleParams = field(default_factory=ViewScaleParams)


def preference_proxy_score(
    actual_likes: float,
    view_count: float,
    hours_since_publish: float,
    config: PreferenceScoreConfig | None = None,
) -> float:
    """Actual likes / likes expected from view scale and comment timing."""
    config = config or PreferenceScoreConfig()
    base_expected_likes = expected_likes_by_views(view_count, config.view_params)
    time_factor = time_advantage(hours_since_publish, config.time_params)
    final_expected = base_expected_likes * time_factor
    return float(actual_likes) / (final_expected + 1e-5)


def _first_present(record: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def score_comment_record(record: dict[str, Any], config: PreferenceScoreConfig | None = None) -> dict[str, Any]:
    """Add `hours_after_publish` and `preference_proxy_score` to a flat comment record."""
    actual_likes = float(_first_present(record, ("comment_like_count", "like_count"), 0))
    view_count = float(_first_present(record, ("video_view_count", "view_count"), 0))

    if "hours_after_publish" in record:
        hours = float(record["hours_after_publish"])
    else:
        video_published = _first_present(record, ("video_published_at", "publishedAt"))
        comment_published = _first_present(record, ("comment_published_at", "published_at"))
        hours = (
            hours_after_publish(str(video_published), str(comment_published))
            if video_published and comment_published else 0.0
        )

    scored = dict(record)
    scored["hours_after_publish"] = hours
    scored["preference_proxy_score"] = preference_proxy_score(actual_likes, view_count, hours, config)
    return scored


def rank_comment_records(
    records: Iterable[dict[str, Any]],
    config: PreferenceScoreConfig | None = None,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    scored = [score_comment_record(record, config) for record in records]
    return sorted(scored, key=lambda item: item["preference_proxy_score"], reverse=reverse)
