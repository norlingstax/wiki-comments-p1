# src/metrics.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_scores(model, X):
    """
    Returns a 1D array of decision scores for the positive class.
    Works for both predict_proba and decision_function models.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()  # in case of binary [n,1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores.ravel()
    raise ValueError("Model has neither predict_proba nor decision_function.")


# --- evaluation
def eval_classification(y_true, y_pred, y_score, labels=("non-toxic","toxic")):
    """
    Compute common metrics and curves. Returns a dict and arrays for plotting.
    """
    rep = classification_report(y_true, y_pred, output_dict=True, target_names=labels)
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    return {
        "report": rep,
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec.tolist(),
        "recall_curve": rec.tolist(),
    }

# --- plotting
def plot_confusion(cm: np.ndarray, labels, out_path: Path, title="Confusion matrix"):
    _ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(4.5, 4))
    ax = plt.gca()
    ax.imshow(cm, interpolation="nearest", cmap="twilight")
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)

    nrows, ncols = cm.shape
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", c="k", linestyle="-", lw=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_aspect("equal")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc(fpr, tpr, roc_auc, out_path: Path, title="ROC curve"):
    _ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", lw=3, c="teal")
    ax.plot([0,1], [0,1], linestyle="--", c="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pr(precision, recall, pr_auc, out_path: Path, title="Precision-Recall curve"):
    _ensure_dir(out_path.parent)
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.2f}", lw=3, c="teal")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
