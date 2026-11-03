"""Build the SFT dataset from collected comments.

Original: "SFT 댓글 추출" and "좋아요 점수 - 관련성 점수" sections of the
"댓글_시간_노출_편향_보정과 SFT" notebook.

Pipeline
  Stage 1  create_candidate_dataset()   (this file, CPU)
           - videos with < 5,000 views are skipped
           - comments exposed < 24 h at collection time are skipped
           - final_score = likes / (ExpectedLikesByViews * TimeAdvantage)
           - comments under 1% of expectation are dropped
           - top 20 per video by final_score
  Stage 2  relevance_scoring/qwen_judge.py  (GPU)
           - LLM judge adds relevance_score (0-100) to every candidate
           - select_final_comments(): sort by (relevance, final_score), keep 10
  Output   [{"video_id", "selected_comments": [{"text", "scores": {...}}]}]
           -> read by training/qwen_omni_lora.py

The CLI also accepts the flat anonymized sample in examples/ (which already
has relevance scores) and runs the scoring + selection on it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.bias_correction.preference_score import PreferenceScoreConfig, preference_proxy_score
from src.bias_correction.time_decay import hours_between
from src.data_collection.youtube_schema import load_json, write_json
from src.preprocessing.comment_cleaner import CommentCleanConfig, clean_text, should_keep_comment


@dataclass(frozen=True)
class CandidateConfig:
    min_views: int = 5000          # MIN_VIEWS
    min_exposure_hours: float = 24.0  # MIN_HOURS
    min_final_score: float = 0.01  # under 1% of expected likes is dropped
    top_n: int = 20                # TOP_N candidates per video


FINAL_SELECTION_SIZE = 10


# ============================================================
# Stage 1: candidates (bias-corrected like score)
# ============================================================

def create_candidate_dataset(
    collection: dict[str, Any],
    config: CandidateConfig | None = None,
    score_config: PreferenceScoreConfig | None = None,
    clean_config: CommentCleanConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or CandidateConfig()
    collected_at = collection["collectedAt"]
    final_dataset = []

    for channel in collection["channels"]:
        for video in channel["videos"]:
            views = int(video["viewCount"])
            if views < config.min_views:
                continue

            candidates = []
            for comment in video["comments"]:
                # Exposure = time the comment had to collect likes
                exposure = hours_between(comment["publishedAt"], collected_at)
                if exposure < config.min_exposure_hours:
                    continue

                text = clean_text(comment["text"], clean_config)
                if not should_keep_comment(text, clean_config):
                    continue

                timing = max(0.1, hours_between(video["publishedAt"], comment["publishedAt"]))
                likes = int(comment["likeCount"])
                final_score = preference_proxy_score(likes, views, timing, score_config)
                if final_score < config.min_final_score:
                    continue

                candidates.append({
                    "comment_id": comment["commentId"],
                    "text": text,
                    "final_score": final_score,
                    "like_count": likes,
                    "video_id": video["videoId"],
                })

            if not candidates:
                continue

            final_dataset.append({
                "video_id": video["videoId"],
                "candidates": sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:config.top_n],
            })

    return final_dataset


# ============================================================
# Stage 2 selection (after the LLM judge adds relevance_score)
# ============================================================

def select_final_comments(scored_candidates: list[dict], top_k: int = FINAL_SELECTION_SIZE) -> list[dict]:
    """Sort by relevance first, then by the bias-corrected final_score."""
    final_selection = sorted(
        scored_candidates,
        key=lambda x: (x.get("relevance_score", 0), x["final_score"]),
        reverse=True,
    )[:top_k]
    return [
        {
            "text": c["text"],
            "scores": {
                "relevance": c.get("relevance_score", 0),
                "final_score": round(c["final_score"], 2),
                "likes": c["like_count"],
            },
        }
        for c in final_selection
    ]


def merge_worker_outputs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Concatenate Score+Relevance_dataset_{1..3}.json into one training file."""
    merged: list[dict[str, Any]] = []
    for path in paths:
        merged.extend(load_json(path))
    return merged


# ============================================================
# Flat sample records (examples/sample_comments.anonymized.json)
# ============================================================

def candidates_from_flat_records(
    records: Iterable[dict[str, Any]],
    score_config: PreferenceScoreConfig | None = None,
    clean_config: CommentCleanConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        text = clean_text(str(r.get("comment_text", "")), clean_config)
        if not should_keep_comment(text, clean_config):
            continue
        timing = max(0.1, float(r.get("hours_after_publish", 0.0)))
        likes = int(r.get("comment_like_count", 0))
        candidate = {
            "text": text,
            "final_score": preference_proxy_score(likes, float(r.get("video_view_count", 0)), timing, score_config),
            "like_count": likes,
            "video_id": str(r.get("video_id")),
        }
        if r.get("relevance_score") is not None:
            candidate["relevance_score"] = float(r["relevance_score"])
        grouped[candidate["video_id"]].append(candidate)
    return grouped


def build_dataset_from_file(input_path: str | Path, output_path: str | Path, top_k: int = FINAL_SELECTION_SIZE) -> list[dict]:
    data = load_json(input_path)

    if isinstance(data, dict) and "channels" in data:
        # Raw collection -> Stage 1 candidates (Stage 2 needs the GPU judge)
        result = create_candidate_dataset(data)
    elif isinstance(data, list):
        grouped = candidates_from_flat_records(data)
        result = [
            {"video_id": vid, "selected_comments": select_final_comments(cands, top_k)}
            for vid, cands in grouped.items()
        ]
    else:
        raise ValueError("Input must be a collection JSON or a list of flat records.")

    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT candidates / selections from comment data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k-per-video", type=int, default=FINAL_SELECTION_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset_from_file(args.input, args.output, args.top_k_per_video)
    print(f"Wrote {len(result)} videos to {args.output}")


if __name__ == "__main__":
    main()
