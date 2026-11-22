"""Selective Auto thinking classifier for Qwen 3 Embedding 0.6B FP16."""

import base64
import math
import re
import struct
from functools import cache

QUERY_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"
MODEL_SIZE = 1_197_629_632
MODEL_SHA256 = "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340"
_DIMENSIONS = 256
_LABELS = ("none", "low", "medium", "high", "dubious")
# Higher margins protect tasks where choosing too little or too much reasoning is costly.
_THRESHOLDS = {
    "none": (0.30, 0.10),
    "low": (0.30, 0.18),
    "medium": (0.38, 0.14),
    "high": (0.48, 0.285),
}
_EFFORT_REQUEST = re.compile(
    r"\b(?:reasoning|thinking|effort|xhigh|budget|chain of thought|"
    r"think (?:(?:really|very|extremely) )?(?:hard|deeply)|"
    r"use (?:none|low|medium|high)(?=\s+(?:for|on|as|when)|[.!?,:]|$)|next level up)\b",
    re.IGNORECASE,
)

# Ridge-4 linear classifier fit to 102 labeled prompts, with gates calibrated on a separate
# development set. Rows are little-endian float32 weights in _LABELS order. The input is the
# normalized 256-dimensional prefix of the official FP16 model's last-token query vector.
_WEIGHTS = (
    "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"
)


def eligible_prompt(prompt):
    """Leave long tasks and explicit effort overrides to the normal Auto selector."""
    return bool(prompt.strip() and len(prompt) <= 320 and not _EFFORT_REQUEST.search(prompt))


@cache
def _weight_rows():
    weights = base64.b64decode(_WEIGHTS)
    return tuple(struct.iter_unpack(f"<{_DIMENSIONS}f", weights))


def classify(vector):
    """Return a confident fast label, or None to let the chat model decide."""
    if len(vector) != _DIMENSIONS:
        return None
    (score, label), (runner_up, _) = sorted(
        (
            (math.sumprod(vector, weights), name)
            for name, weights in zip(_LABELS, _weight_rows(), strict=True)
        ),
        reverse=True,
    )[:2]
    if thresholds := _THRESHOLDS.get(label):
        minimum, margin = thresholds
        if score >= minimum and score - runner_up >= margin:
            return label
    return None
