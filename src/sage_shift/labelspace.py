"""The six-class HyperKvasir -> SAGE label space and its meaning groups.

The mapping reproduces `HYPERKVASIR_2_sage` in the SAGE authors' `hyperkvasir-on-SAGE.py`
(lines 80-104, commit e24eb0e). HyperKvasir spells eight class names differently in its official
split files than in its `image-labels.csv`; both spellings are accepted, and unknown names raise.
"""

from __future__ import annotations

# The authors fit a LabelEncoder on the mapped names, which orders them alphabetically
# (hyperkvasir-on-SAGE.py:118-119). The same order is used here.
SIX_CLASSES: tuple[str, ...] = (
    "esophagitis",
    "fundus-retroflexion",
    "polyp",
    "pylorus-normal",
    "rectum-retroflexion",
    "z-line-normal",
)

# The SAGE multi-label name behind each class (the `class` value without -normal / -retroflexion).
BASE_LABEL: dict[str, str] = {
    "esophagitis": "esophagitis",
    "fundus-retroflexion": "fundus",
    "polyp": "polyp",
    "pylorus-normal": "pylorus",
    "rectum-retroflexion": "rectum",
    "z-line-normal": "z-line",
}

# Do the HyperKvasir and SAGE classes share their meaning, or only their name?
MEANING_GROUP: dict[str, str] = {
    "esophagitis": "matched",  # HyperKvasir grades A and B-D merged; SAGE records no grade
    "fundus-retroflexion": "matched",
    "rectum-retroflexion": "matched",
    "z-line-normal": "matched",
    "polyp": "shifted",  # HyperKvasir polyps are colonic; SAGE's are mostly gastric
    "pylorus-normal": "shifted",  # SAGE's pylorus label also absorbs the antrum tag
}

# HyperKvasir `Finding` names (image-labels.csv) -> SAGE class, or None where the authors dropped it.
HK_FINDING_TO_SAGE: dict[str, str | None] = {
    "cecum": None,
    "ileum": None,
    "retroflex-rectum": "rectum-retroflexion",
    "hemorrhoids": None,
    "polyps": "polyp",
    "ulcerative-colitis-grade-0-1": None,
    "ulcerative-colitis-grade-1": None,
    "ulcerative-colitis-grade-1-2": None,
    "ulcerative-colitis-grade-2": None,
    "ulcerative-colitis-grade-2-3": None,
    "ulcerative-colitis-grade-3": None,
    "bbps-0-1": None,
    "bbps-2-3": None,
    "impacted-stool": None,
    "dyed-lifted-polyps": None,
    "dyed-resection-margins": None,
    "pylorus": "pylorus-normal",
    "retroflex-stomach": "fundus-retroflexion",
    "z-line": "z-line-normal",
    "barretts": None,
    "barretts-short-segment": None,
    "esophagitis-a": "esophagitis",
    "esophagitis-b-d": "esophagitis",
}

# Spellings in HyperKvasir's official split files (official_splits/*.csv) that differ from the above.
HK_SPLIT_SPELLING: dict[str, str] = {
    "normal-cecum": "cecum",
    "normal-pylorus": "pylorus",
    "normal-z-line": "z-line",
    "oesophagitis-a": "esophagitis-a",
    "oesophagitis-b-d": "esophagitis-b-d",
    "polyp": "polyps",
    "hemorroids": "hemorrhoids",
    "short-segment-barretts": "barretts-short-segment",
}


def hk_finding(name: str) -> str:
    """Return the image-labels.csv spelling of a HyperKvasir class given either spelling."""
    finding = HK_SPLIT_SPELLING.get(name, name)
    if finding not in HK_FINDING_TO_SAGE:
        raise KeyError(f"unknown HyperKvasir class name: {name!r}")
    return finding


def hk_to_sage(name: str) -> str | None:
    """SAGE class for a HyperKvasir class (either spelling), or None if it is outside the label space."""
    return HK_FINDING_TO_SAGE[hk_finding(name)]
