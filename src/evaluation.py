"""
Evaluation module for the Medical RAG QA system.
Computes retrieval and generation metrics, and produces visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from typing import List, Dict, Optional


# ── Classification Metrics ──────────────────────────────────────────

def compute_classification_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str] = None
) -> dict:
    """Compute accuracy, precision, recall, F1 for yes/no/maybe classification."""
    if labels is None:
        labels = ["yes", "no", "maybe"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {
            label: {"precision": p, "recall": r, "f1": f, "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


# ── Retrieval Metrics ───────────────────────────────────────────────

def recall_at_k(relevant_doc_idx: int, retrieved_indices: List[int], k: int) -> float:
    """Binary recall@K: 1 if the relevant doc is in top-K retrieved, else 0."""
    return 1.0 if relevant_doc_idx in retrieved_indices[:k] else 0.0


def mean_reciprocal_rank(relevant_idx: int, retrieved_indices: List[int]) -> float:
    """MRR for a single query."""
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    queries_results: List[Dict], k_values: List[int] = None
) -> dict:
    """
    Compute retrieval metrics over a set of queries.

    Args:
        queries_results: List of dicts with 'relevant_idx' and 'retrieved_indices'
        k_values: List of K values for recall@K
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    recall_scores = {k: [] for k in k_values}
    mrr_scores = []

    for qr in queries_results:
        rel_idx = qr["relevant_idx"]
        ret_indices = qr["retrieved_indices"]

        mrr_scores.append(mean_reciprocal_rank(rel_idx, ret_indices))
        for k in k_values:
            recall_scores[k].append(recall_at_k(rel_idx, ret_indices, k))

    return {
        "mrr": np.mean(mrr_scores),
        **{f"recall@{k}": np.mean(recall_scores[k]) for k in k_values},
    }


# ── Visualizations ──────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot a confusion matrix heatmap."""
    if labels is None:
        labels = ["yes", "no", "maybe"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_retrieval_recall(
    recall_scores: dict,
    title: str = "Recall@K",
    save_path: Optional[str] = None,
):
    """Plot recall@K curve."""
    ks = sorted(recall_scores.keys())
    scores = [recall_scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("K (number of retrieved documents)")
    ax.set_ylabel("Recall@K")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_embedding_clusters(
    embeddings: np.ndarray,
    labels: List[str],
    title: str = "Document Embeddings (UMAP)",
    save_path: Optional[str] = None,
):
    """Plot UMAP projection of embeddings colored by label."""
    import umap

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique_labels))

    for label, color in zip(unique_labels, colors):
        mask = [l == label for l in labels]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], label=label, alpha=0.6, s=20,
        )

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_answer_comparison(
    examples: List[Dict],
    save_path: Optional[str] = None,
):
    """
    Create a qualitative comparison table of RAG vs ground truth answers.

    Args:
        examples: List of dicts with 'question', 'true_answer', 'pred_answer',
                  'true_decision', 'pred_decision'
    """
    fig, ax = plt.subplots(figsize=(14, 2 + len(examples) * 2))
    ax.axis("off")

    cell_text = []
    for ex in examples:
        cell_text.append([
            ex["question"][:80] + "..." if len(ex["question"]) > 80 else ex["question"],
            ex["true_decision"],
            ex["pred_decision"],
            "Match" if ex["true_decision"] == ex["pred_decision"] else "Mismatch",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Question", "True", "Predicted", "Match"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color match/mismatch cells
    for i, row in enumerate(cell_text, 1):
        color = "#c8e6c9" if row[3] == "Match" else "#ffcdd2"
        table[i, 3].set_facecolor(color)

    ax.set_title("RAG Answer Comparison: Predicted vs Ground Truth", fontsize=12, pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
