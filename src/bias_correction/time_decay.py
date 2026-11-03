"""Time-decay correction for comment likes.

Original: "댓글_시간_노출_편향_보정과 SFT" notebook, cells
  - analyze_with_modeling(): curve fitting of the decay model
  - create_candidate_dataset(): decay_model(timing, *DECAY_PARAMS)

Model
    TimeAdvantage(t) = a * t^(-b) + c
    t = hours between video upload and comment, clamped to >= 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class TimeDecayParams:
    """Parameters fitted in the project (DECAY_PARAMS = (1.65, 0.53, 0.08))."""

    a: float = 1.65
    b: float = 0.53
    c: float = 0.08
    min_hours: float = 0.1


def decay_model(x, a, b, c):
    """y = a * x^(-b) + c : high right after upload, converges to c over time."""
    return a * np.power(x, -b) + c


def parse_youtube_datetime(value: str) -> datetime:
    """Parse a YouTube ISO timestamp into a timezone-aware UTC datetime."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hours_between(start: str, end: str) -> float:
    return (parse_youtube_datetime(end) - parse_youtube_datetime(start)).total_seconds() / 3600


def hours_after_publish(video_published_at: str, comment_published_at: str) -> float:
    """Hours from video upload to comment, never negative."""
    return max(hours_between(video_published_at, comment_published_at), 0.0)


def time_advantage(hours: float, params: TimeDecayParams | None = None) -> float:
    """Expected like-efficiency for a comment written `hours` after upload."""
    params = params or TimeDecayParams()
    t = max(float(hours), params.min_hours)
    return float(decay_model(t, params.a, params.b, params.c))


def correct_likes_by_time(actual_likes: float, hours: float, params: TimeDecayParams | None = None) -> float:
    return float(actual_likes) / (time_advantage(hours, params) + 1e-5)


# ------------------------------------------------------------------
# Fitting (how a, b, c were obtained)
# ------------------------------------------------------------------

def build_decay_fit_rows(collection: dict, min_views: int = 1000, min_hours: float = 0.1) -> list[dict]:
    """One row per comment: hours after upload and likes per 100k views.

    Videos under 1,000 views are dropped as noise, and comments written within
    0.1 h are dropped to keep the power law numerically stable.
    """
    rows = []
    for channel in collection["channels"]:
        for video in channel["videos"]:
            v_views = int(video["viewCount"])
            if v_views < min_views:
                continue
            for comment in video["comments"]:
                hours_diff = hours_between(video["publishedAt"], comment["publishedAt"])
                likes = int(comment["likeCount"])
                if hours_diff > min_hours:
                    rows.append({
                        "video_id": video["videoId"],
                        "text": comment["text"],
                        "hours_diff": hours_diff,
                        "norm_likes": (likes / v_views) * 100000,
                        "raw_likes": likes,
                    })
    return rows


def fit_time_decay(hours: np.ndarray, norm_likes: np.ndarray) -> TimeDecayParams:
    """Non-linear least squares fit of decay_model, p0 = (10, 0.5, 0.1)."""
    from scipy.optimize import curve_fit

    try:
        popt, _ = curve_fit(decay_model, hours, norm_likes, p0=[10, 0.5, 0.1], maxfev=5000)
        a_opt, b_opt, c_opt = popt
    except Exception:
        print("Curve fitting failed - using defaults")
        a_opt, b_opt, c_opt = 10, 0.5, 0.1
    return TimeDecayParams(a=float(a_opt), b=float(b_opt), c=float(c_opt))


def performance_scores(hours: np.ndarray, norm_likes: np.ndarray, params: TimeDecayParams) -> np.ndarray:
    """Observed / expected (O/E). 1.0 = average for that timing."""
    expected = decay_model(hours, params.a, params.b, params.c)
    return norm_likes / (expected + 1e-5)
