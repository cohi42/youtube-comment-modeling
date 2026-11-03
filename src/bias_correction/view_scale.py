"""View-scale correction for comment likes.

Original: "댓글_시간_노출_편향_보정과 SFT" notebook, cells
  - analyze_power_law(): log-log linear regression
  - create_candidate_dataset(): scale_model(views, *SCALE_PARAMS)

Model (power law)
    ExpectedLikes = 10^beta * Views^alpha
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewScaleParams:
    """Parameters fitted in the project (SCALE_PARAMS = (1.0578, -3.4334))."""

    alpha: float = 1.0578
    beta: float = -3.4334
    min_views: int = 1


def scale_model(views, alpha, beta):
    """Expected likes for a video with `views` views."""
    return (10 ** beta) * np.power(views, alpha)


def expected_likes_by_views(view_count: float, params: ViewScaleParams | None = None) -> float:
    params = params or ViewScaleParams()
    views = max(float(view_count), params.min_views)
    return float(scale_model(views, params.alpha, params.beta))


def correct_likes_by_views(actual_likes: float, view_count: float, params: ViewScaleParams | None = None) -> float:
    return float(actual_likes) / (expected_likes_by_views(view_count, params) + 1e-5)


# ------------------------------------------------------------------
# Fitting (how alpha, beta were obtained)
# ------------------------------------------------------------------

def build_scale_fit_rows(collection: dict, min_views: int = 1000) -> list[dict]:
    """One row per video: views and the like count of its best comment.

    The max is used to measure each video's potential. Videos with <= 1,000
    views or a best comment with 0 likes are dropped.
    """
    records = []
    for channel in collection["channels"]:
        for video in channel["videos"]:
            if not video["comments"]:
                continue
            views = int(video["viewCount"])
            max_likes = max(int(c["likeCount"]) for c in video["comments"])
            if views > min_views and max_likes > 0:
                records.append({"viewCount": views, "maxLikes": max_likes})
    return records


def fit_view_scale(view_counts: np.ndarray, max_likes: np.ndarray) -> tuple[ViewScaleParams, float]:
    """Linear regression on log10(views) -> log10(max likes). Returns (params, R^2)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    x_log = np.log10(np.asarray(view_counts).reshape(-1, 1))
    y_log = np.log10(np.asarray(max_likes))

    model = LinearRegression()
    model.fit(x_log, y_log)

    alpha = float(model.coef_[0])     # slope -> scale factor
    beta = float(model.intercept_)    # intercept -> baseline
    r2 = float(r2_score(y_log, model.predict(x_log)))
    return ViewScaleParams(alpha=alpha, beta=beta), r2
