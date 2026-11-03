from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any


KOREAN_RE = re.compile(r"[\uac00-\ud7af]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CommentCleanConfig:
    """Rules for filtering comments before scoring or training."""

    min_length: int = 2
    max_length: int = 300
    max_repeated_chars: int = 4
    require_korean: bool = True
    blocked_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "구독",
            "좋아요 부탁",
            "광고",
            "이벤트",
            "고정해주세요",
            "고정 좀",
            "http",
            "www.",
        )
    )


def has_korean(text: str) -> bool:
    return bool(KOREAN_RE.search(text or ""))


def normalize_repeated_chars(text: str, max_repeated_chars: int = 4) -> str:
    """Limit long character runs while preserving Korean comment style."""

    if max_repeated_chars < 1:
        return text

    pattern = re.compile(r"(.)\1{" + str(max_repeated_chars) + r",}")
    return pattern.sub(lambda match: match.group(1) * max_repeated_chars, text)


def mask_private_tokens(text: str) -> str:
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


def clean_text(text: str, config: CommentCleanConfig | None = None) -> str:
    config = config or CommentCleanConfig()
    text = html.unescape(text or "")
    text = HTML_TAG_RE.sub(" ", text)
    text = mask_private_tokens(text)
    text = normalize_repeated_chars(text, config.max_repeated_chars)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def should_keep_comment(text: str, config: CommentCleanConfig | None = None) -> bool:
    config = config or CommentCleanConfig()
    normalized = clean_text(text, config)

    if len(normalized) < config.min_length:
        return False
    if len(normalized) > config.max_length:
        return False
    if config.require_korean and not has_korean(normalized):
        return False

    lowered = normalized.lower()
    if any(keyword.lower() in lowered for keyword in config.blocked_keywords):
        return False

    return True


def clean_comment_record(
    record: dict[str, Any],
    text_key: str = "comment_text",
    config: CommentCleanConfig | None = None,
) -> dict[str, Any] | None:
    """Return a cleaned copy of a comment record or None if filtered out."""

    config = config or CommentCleanConfig()
    text = clean_text(str(record.get(text_key, "")), config)

    if not should_keep_comment(text, config):
        return None

    cleaned = dict(record)
    cleaned[text_key] = text
    return cleaned

