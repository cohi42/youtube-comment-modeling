"""Prompts for the multimodal LLM judge.

Two different judges were used in the project:

1. Relevance judge (dataset building)
   Original: QwenJudge.evaluate() in the "댓글_시간_노출_편향_보정과 SFT" notebook.
   Scores up to 10 candidate comments at once (0-100) for how well they fit the
   video's context, vibe and topic. Used to filter SFT training targets.

2. Generation evaluation judge (experiment results)
   Scores one generated comment on Contextual Grounding / Semantic Resonance /
   Human-like Tone. Used for the result tables in docs/experiments.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ============================================================
# 1. Relevance judge (dataset building)
# ============================================================

RELEVANCE_CHUNK_SIZE = 10
RELEVANCE_COMMENT_MAX_CHARS = 100


def format_comment_chunk(comments: list[str]) -> str:
    """'[1] text\\n[2] text\\n...' with each comment cut to 100 chars on one line."""
    chunk_text = ""
    for idx, text in enumerate(comments):
        safe_text = text[:RELEVANCE_COMMENT_MAX_CHARS].replace("\n", " ")
        chunk_text += f"[{idx + 1}] {safe_text}\n"
    return chunk_text


def build_relevance_judge_prompt(comments: list[str]) -> str:
    """Prompt sent together with the video's frames and audio."""
    n = len(comments)
    chunk_text = format_comment_chunk(comments)
    return (
        "You are an AI Analyst specialized in 'Semantic & Cultural Context' of video comments.\n"
        "Your task is to evaluate how well the comments align with the video's **context, vibe, and topic** (0-100).\n\n"

        "⚠️ CRITICAL INSTRUCTIONS:\n"
        f"1. PROCESS ALL: You must generate scores for ALL {n} comments. Do not stop halfway.\n"
        "2. ANTI-ROUNDING: Do NOT use numbers ending in 0 or 5 (e.g., 87, 12, 93 instead of 80, 90).\n"
        "3. **SEMANTIC MATCH**: Do not be too literal. If the video shows a 'GPS' and the comment says 'Map', "
        "that is a HIGH match. Look for metaphorical connections, emotional resonance, and broad topical relevance.\n\n"

        "Scoring Criteria (0-100):\n"
        "- 0-19 (None): Spam, completely unrelated, generic greetings.\n"
        "- 20-39 (Weak): Vague reactions (e.g., 'lol', 'wow') applicable to any video.\n"
        "- 40-59 (Moderate): Broadly related to the genre/topic but generic (e.g., talking about 'driving' generally).\n"
        "- 60-79 (Semantic): Matches the **meaning or vibe**. Metaphors, related concepts, or emotional reactions to specific scenes.\n"
        "- 80-100 (Perfect): Sharp insight, witty observation about a specific moment, or strong empathy with the video's situation.\n\n"

        f"--- Comments to Evaluate ({n} items) ---\n{chunk_text}\n"

        "--- Output Format ---\n"
        "Format: [ID] Reason: (Brief semantic connection) -> Score: N\n\n"
        "Example Output:\n"
        "[1] Reason: Calls the GPS a 'paper map' metaphorically, fitting the getting lost context. -> Score: 88\n"
        "[2] Reason: Just a generic laugh 'kkk'. -> Score: 23\n"
        "[3] Reason: Mentions the background music vibe accurately. -> Score: 94\n"
        "... (continue for all items)\n\n"
        "Start processing from [1]:"
    )


SCORE_LINE_RE = re.compile(r"\[(\d+)\][^S]*Score:\s*(\d+)")


def parse_relevance_scores(judge_output: str, n_comments: int) -> list[float]:
    """Map '[ID] ... Score: N' lines back to the chunk. Missing IDs get 0."""
    score_map = {int(idx) - 1: float(score) for idx, score in SCORE_LINE_RE.findall(judge_output)}
    return [score_map.get(j, 0.0) for j in range(n_comments)]


# ============================================================
# 2. Generation evaluation judge (experiment results)
# ============================================================

@dataclass(frozen=True)
class JudgeCriteria:
    contextual_grounding: str = "댓글이 영상의 시각적 또는 청각적 정보에 기반해 있는지 평가한다."
    semantic_resonance: str = "댓글이 영상의 분위기, 재미 요소, 음식 도메인의 공감 포인트를 반영하는지 평가한다."
    human_like_tone: str = "댓글이 실제 한국 YouTube 댓글창에서 볼 법한 자연스러운 말투인지 평가한다."


DEFAULT_JUDGE_CRITERIA = JudgeCriteria()


def build_generation_eval_prompt(
    generated_comment: str,
    criteria: JudgeCriteria = DEFAULT_JUDGE_CRITERIA,
    score_scale: str = "0-100",
) -> str:
    """Prompt for scoring one generated comment; the video is attached separately."""
    return f"""
당신은 'Youtube Comment Judge (유튜브 댓글 심사위원)'입니다.
입력받은 영상을 바탕으로, AI 모델이 생성한 댓글이 얼마나 적절한지 평가하세요.
점수는 각 항목별로 {score_scale} 범위의 정수로 주세요.

[평가 기준]
1. Contextual Grounding (영상 내용 일치도)
- {criteria.contextual_grounding}
- 시각과 청각을 모두 언급할 필요는 없습니다. 댓글이 집중한 포인트가 영상과 일치하면 됩니다.
- 영상에 없는 내용을 꾸며내거나 영상과 모순되면 최저점을 주세요.

2. Semantic Resonance (의미적 적합성)
- {criteria.semantic_resonance}
- 단순한 사실 묘사보다 영상의 포인트를 짚은 댓글에 더 높은 점수를 주세요.

3. Human-like Tone (인간적 문체)
- {criteria.human_like_tone}
- 번역투, 지나치게 격식적인 문장, AI 요약문처럼 보이는 문장은 감점하세요.

[평가할 댓글]
{generated_comment}

[출력 형식]
JSON만 출력하세요.
{{
  "scores": {{
    "contextual_grounding": <int>,
    "semantic_resonance": <int>,
    "human_like_tone": <int>
  }},
  "reasoning": "각 점수에 대한 구체적인 근거"
}}
""".strip()


def extract_eval_score(judge_result: dict) -> float:
    """FinalScore = mean of the three axes."""
    scores = judge_result.get("scores", {})
    if "average_score" in scores:
        return float(scores["average_score"])
    values = [
        scores[key]
        for key in ("contextual_grounding", "semantic_resonance", "human_like_tone")
        if key in scores
    ]
    return float(sum(values) / len(values)) if values else 0.0
