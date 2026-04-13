"""
Data utilities for loading and filtering PubMedQA dataset.
Focuses on hormonal health, reproductive health, and women's health domains.
"""

from datasets import load_dataset
import pandas as pd
from typing import Optional


# Keywords for filtering to women's/hormonal health domain
HEALTH_KEYWORDS = [
    # Hormonal conditions
    "pcos", "polycystic ovary", "polycystic ovarian",
    "endometriosis", "endometrial",
    "amenorrhea", "dysmenorrhea", "menorrhagia",
    "premenstrual", "pms", "pmdd",
    "perimenopause", "menopause", "postmenopause",
    "hirsutism", "hyperandrogenism",
    # Hormones
    "estrogen", "oestrogen", "progesterone", "testosterone",
    "luteinizing hormone", "follicle stimulating", "fsh", "lh",
    "prolactin", "cortisol", "thyroid",
    "hormonal", "hormone",
    # Reproductive health
    "ovulation", "ovulatory", "anovulation",
    "fertility", "infertility", "subfertility",
    "menstrual cycle", "menstruation", "menses",
    "uterine", "uterus", "ovarian", "ovary",
    "contracepti", "oral contraceptive",
    "pregnancy", "pregnant", "prenatal", "postnatal",
    "breastfeeding", "lactation",
    # Women's health general
    "women's health", "female health",
    "gynecol", "obstetric",
    "cervical", "breast cancer",
    "osteoporosis",
    "iron deficiency", "anemia",
]


def load_pubmedqa(subset: str = "pqa_labeled") -> dict:
    """
    Load PubMedQA dataset from HuggingFace.

    Args:
        subset: One of 'pqa_labeled' (1K expert), 'pqa_artificial' (211K), 'pqa_unlabeled'

    Returns:
        Dataset dict
    """
    dataset = load_dataset("qiaojin/PubMedQA", subset, trust_remote_code=True)
    return dataset


def is_health_related(text: str, keywords: list = HEALTH_KEYWORDS) -> bool:
    """Check if text contains any of the health-related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def filter_health_domain(dataset, fields: list = None) -> list:
    """
    Filter dataset to women's/hormonal health domain.

    Args:
        dataset: HuggingFace dataset split
        fields: Fields to search for keywords. Defaults to question + context.

    Returns:
        List of filtered examples
    """
    if fields is None:
        fields = ["question", "long_answer"]

    filtered = []
    for example in dataset:
        searchable = " ".join(str(example.get(f, "")) for f in fields)
        # Also search in context (list of strings)
        if "context" in example and isinstance(example["context"], dict):
            contexts = example["context"].get("contexts", [])
            searchable += " " + " ".join(contexts)
        if is_health_related(searchable):
            filtered.append(example)

    return filtered


def dataset_to_dataframe(examples: list) -> pd.DataFrame:
    """Convert list of dataset examples to a pandas DataFrame for analysis."""
    rows = []
    for ex in examples:
        context_texts = []
        if "context" in ex and isinstance(ex["context"], dict):
            context_texts = ex["context"].get("contexts", [])

        rows.append({
            "pubid": ex.get("pubid", ""),
            "question": ex.get("question", ""),
            "context": " ".join(context_texts),
            "long_answer": ex.get("long_answer", ""),
            "final_decision": ex.get("final_decision", ""),
            "num_contexts": len(context_texts),
        })
    return pd.DataFrame(rows)


def get_split_stats(df: pd.DataFrame) -> dict:
    """Get basic statistics about the dataset."""
    stats = {
        "total_examples": len(df),
        "label_distribution": df["final_decision"].value_counts().to_dict() if "final_decision" in df.columns else {},
        "avg_question_length": df["question"].str.len().mean(),
        "avg_context_length": df["context"].str.len().mean(),
        "avg_answer_length": df["long_answer"].str.len().mean(),
    }
    return stats
