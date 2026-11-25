"""SAGE audit section 6: captions, VQA pairs, question templates and visibility answers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from sage_shift.audit_types import SPLIT_KEYS, Audit, SageData, by_split, count_by_split, pct

INLINE_TAG = re.compile(r"</?\s*[A-Za-z][A-Za-z0-9]*\s*>")


def _caption_flags(data: SageData) -> pd.DataFrame:
    """One row of counts and flags per caption (no caption text is kept)."""
    split_of = data.samples.set_index("sample_uuid")["split"]
    rows = []
    for sample_uuid, caption in data.captions.items():
        spans = caption["annotation"]
        draft, corrected = caption["gpt_description"], caption["corrected_description"]
        rows.append(
            {
                "split": split_of.get(sample_uuid),
                "n_spans": len(spans),
                "n_hallucinated": sum(bool(span["hallucinated"]) for span in spans),
                "n_omission": sum(bool(span["omission"]) for span in spans),
                "has_inline_tag": bool(INLINE_TAG.search(draft + corrected)),
                "unchanged": draft == corrected,
                "spans_rebuild_draft": "".join(span["text"] for span in spans) == draft,
            }
        )
    return pd.DataFrame(rows)


def audit_text(data: SageData, audit: Audit, cfg: dict[str, Any]) -> None:
    """Section 6: caption and VQA counts, question templates and visibility answers."""
    section = "6 Text"
    expected = cfg["expected"]
    captions = _caption_flags(data)
    metrics = {
        "captions": lambda part: len(part),
        "captions with >=1 hallucinated span": lambda part: int((part["n_hallucinated"] > 0).sum()),
        "captions with >=1 omission span": lambda part: int((part["n_omission"] > 0).sum()),
        "captions unchanged by the expert": lambda part: int(part["unchanged"].sum()),
        "captions with inline <tag> markers": lambda part: int(part["has_inline_tag"].sum()),
        "annotation spans": lambda part: int(part["n_spans"].sum()),
        "hallucinated spans": lambda part: int(part["n_hallucinated"].sum()),
        "omission spans": lambda part: int(part["n_omission"].sum()),
    }
    caption_summary = {name: by_split(captions, func) for name, func in metrics.items()}
    audit.tables["captions_summary"] = pd.DataFrame(
        [{"metric": name, **values} for name, values in caption_summary.items()]
    )
    rebuilt = int(captions["spans_rebuild_draft"].sum())
    audit.check(
        section,
        "annotation spans concatenate to gpt_description for every caption",
        rebuilt == len(captions),
        f"{rebuilt} of {len(captions)} captions",
    )
    tagged = caption_summary["captions with inline <tag> markers"]["total"]
    hallucinated = caption_summary["captions with >=1 hallucinated span"]["total"]
    audit.discrepancies.append(
        f"Hallucination markers: the paper's Figure 1 marks hallucinations with inline tags such as `<H>…</H>`; "
        f"{tagged} captions contain any inline tag. As the README describes, hallucinations are stored as "
        f"`annotation` spans with a `hallucinated` flag ({hallucinated} of {len(captions)} captions have at least one)."
    )

    split_of = data.samples.set_index("sample_uuid")["split"]
    pairs = pd.DataFrame(
        [
            {"sample_uuid": sample_uuid, "question": pair["question"], "answer": pair["answer"]}
            for sample_uuid, items in data.vqa.items()
            for pair in items
        ]
    )
    pairs["split"] = pairs["sample_uuid"].map(split_of)
    n_images = by_split(data.samples, len)
    questions = count_by_split(pairs["question"], pairs["split"]).loc[pairs["question"].drop_duplicates()]
    images_with = pairs.groupby("question")["sample_uuid"].nunique()
    questions["images_without_this_question"] = n_images["total"] - images_with
    audit.tables["vqa_questions"] = questions.rename_axis("question").reset_index()
    per_image = pairs.groupby("sample_uuid").size().reindex(data.samples["sample_uuid"], fill_value=0)
    pairs_per_image = {int(key): int(value) for key, value in per_image.value_counts().sort_index().items()}

    audit.check(
        section,
        f"VQA pair count matches the paper ({expected['n_vqa_pairs']:,})",
        len(pairs) == expected["n_vqa_pairs"],
        f"{len(pairs):,} pairs",
    )
    paper_questions = expected["n_vqa_questions_paper"]
    audit.check(
        section,
        f"number of VQA question templates matches the paper ({paper_questions})",
        len(questions) == paper_questions,
        f"{len(questions)} distinct question strings in vqa.json",
    )
    if len(questions) != paper_questions:
        audit.discrepancies.append(
            f"VQA questions: the paper describes {paper_questions} questions; vqa.json has {len(questions)} distinct "
            f"question templates. Pairs per image: {pairs_per_image}."
        )
    repeated = int(pairs.duplicated(["sample_uuid", "question"]).sum())
    empty_answers = int((pairs["answer"].str.strip() == "").sum())
    audit.check(
        section,
        "no image repeats a question and no answer is empty",
        repeated == 0 and empty_answers == 0,
        f"{repeated} repeated questions, {empty_answers} empty answers",
    )

    visibility = _visibility_summary(pairs, n_images, audit, cfg)
    audit.summary["text"] = {
        "captions": caption_summary,
        "n_vqa_pairs": by_split(pairs, len),
        "n_question_templates": len(questions),
        "pairs_per_image": pairs_per_image,
        "visibility": visibility,
    }


def _visibility_summary(
    pairs: pd.DataFrame, n_images: dict[str, int], audit: Audit, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Describe the free-text visibility answers as they are, without binning them."""
    prefix = cfg["visibility_question_prefix"]
    matching = [question for question in pairs["question"].unique() if question.startswith(prefix)]
    if len(matching) != 1:
        return {"found": False, "n_matching_questions": len(matching)}
    answers = pairs[pairs["question"] == matching[0]]
    pattern = re.compile(re.escape(cfg["visibility_answer_stem"]) + r"\s+([A-Za-z-]+)")

    def leading_word(answer: str) -> str:
        match = pattern.match(answer.strip())
        return match.group(1).lower() if match else "(other phrasing)"

    words = answers["answer"].map(leading_word)
    leading = count_by_split(words, answers["split"])
    n_answers = by_split(answers, len)
    leading.loc["(question absent for the image)"] = [n_images[key] - n_answers[key] for key in SPLIT_KEYS]
    leading = leading.sort_values("total", ascending=False)
    for key in SPLIT_KEYS:
        leading[f"{key}_pct"] = [pct(value, n_images[key]) for value in leading[key]]
    audit.tables["vqa_visibility_leading_word"] = leading.rename_axis("word_after_stem").reset_index()
    audit.tables["vqa_visibility_top_answers"] = (
        count_by_split(answers["answer"], answers["split"])
        .sort_values("total", ascending=False)
        .head(15)
        .rename_axis("answer_verbatim")
        .reset_index()
    )
    return {
        "found": True,
        "question": matching[0],
        "n_answers": n_answers,
        "n_distinct_answers": int(answers["answer"].nunique()),
        "answer_stem": cfg["visibility_answer_stem"],
        "n_answers_starting_with_stem": int((words != "(other phrasing)").sum()),
        "n_distinct_leading_words": int(words[words != "(other phrasing)"].nunique()),
    }
