"""Stratification flags for SAGE images from labels, captions and VQA answers.

Used only for subgroup and sensitivity analyses, never as prediction targets. The text rules are
simple keyword matches with clause-level negation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

VISIBILITY_PREFIX = "What is the visibility in the image?"
VISIBILITY_WORDS: dict[str, frozenset[str]] = {
    "good": frozenset({"good", "clear"}),
    "fair": frozenset({"moderate", "limited", "reduced", "partial", "partially", "fair"}),
    "poor": frozenset({"poor", "not", "no", "entirely", "obstructed"}),
}
UPPER_GI_LABELS = frozenset({"esophagus", "gej", "z-line", "stomach", "fundus", "pylorus", "duodenum"})
LOWER_GI_LABELS = frozenset({"colon", "rectum"})

_NEGATION = re.compile(r"\b(no|not|none|without|absent|neither|nor)\b")
_CLAUSE_BREAK = re.compile(r"[,;:]|\bbut\b|\bwhile\b|\bwhereas\b|\bhowever\b")
_VISIBILITY_WORD = re.compile(r"visibility in the image is\s+(\w+)")
_NBI = r"narrow[- ]band|\bnbi\b"


def sentences(text: str) -> list[str]:
    """Split text into lower-case sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", str(text).lower()) if s.strip()]


def affirms(text: str, pattern: str) -> bool:
    """True if a clause matches `pattern` with no negation word before the match in that clause.

    "The endoscope is visible, but it is not targeting anything" affirms "endoscop";
    "There is no endoscope" does not.
    """
    regex = re.compile(pattern)
    for sentence in sentences(text):
        for clause in _CLAUSE_BREAK.split(sentence):
            match = regex.search(clause)
            if match and not _NEGATION.search(clause[: match.start()]):
                return True
    return False


def vqa_answer(pairs: Sequence[Mapping[str, str]], prefix: str) -> str:
    """The answer to the first question that starts with `prefix`, or "" if there is none."""
    for pair in pairs:
        if str(pair.get("question", "")).startswith(prefix):
            return str(pair.get("answer", ""))
    return ""


def visibility_bin(answer: str) -> str:
    """Bin a VQA visibility answer by the word after "The visibility in the image is".

    Returns good, fair or poor; "unbinned" for other wordings and "missing" for an empty answer.
    """
    if not str(answer).strip():
        return "missing"
    match = _VISIBILITY_WORD.search(str(answer).lower())
    word = match.group(1) if match else ""
    for name, words in VISIBILITY_WORDS.items():
        if word in words:
            return name
    return "unbinned"


def nbi_flag(caption: str) -> bool:
    """True if the caption affirms narrow-band imaging."""
    return affirms(caption, _NBI)


def location_context(labels: Iterable[str]) -> str:
    """Where an image's location labels place it: upper GI, lower GI, both or no location label."""
    found = set(labels)
    upper, lower = bool(found & UPPER_GI_LABELS), bool(found & LOWER_GI_LABELS)
    if upper and lower:
        return "both"
    if upper:
        return "upper GI"
    if lower:
        return "lower GI"
    return "no location label"
