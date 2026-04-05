from __future__ import annotations

import numpy as np


def off_diagonal_mask(n: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def confusion_counts(truth: np.ndarray, predicted: np.ndarray, *, mask: np.ndarray | None = None) -> dict[str, int]:
    truth_flat, predicted_flat = _flatten_pair(truth, predicted, mask)
    tp = int(np.count_nonzero(truth_flat & predicted_flat))
    tn = int(np.count_nonzero(~truth_flat & ~predicted_flat))
    fp = int(np.count_nonzero(~truth_flat & predicted_flat))
    fn = int(np.count_nonzero(truth_flat & ~predicted_flat))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(truth: np.ndarray, predicted: np.ndarray, *, mask: np.ndarray | None = None) -> dict[str, float]:
    counts = confusion_counts(truth, predicted, mask=mask)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0
    false_discovery_rate = fp / (tp + fp) if (tp + fp) else 0.0

    return {
        **counts,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "false_discovery_rate": false_discovery_rate,
    }


def auprc(truth: np.ndarray, scores: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    truth_flat, score_flat = _flatten_pair(truth, scores, mask)
    positives = np.count_nonzero(truth_flat)
    if positives == 0:
        return float("nan")

    order = np.argsort(-score_flat)
    truth_sorted = truth_flat[order].astype(int)
    tp = np.cumsum(truth_sorted)
    fp = np.cumsum(1 - truth_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0]], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def auroc(truth: np.ndarray, scores: np.ndarray, *, mask: np.ndarray | None = None) -> float:
    truth_flat, score_flat = _flatten_pair(truth, scores, mask)
    positives = np.count_nonzero(truth_flat)
    negatives = truth_flat.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-score_flat)
    truth_sorted = truth_flat[order].astype(int)
    tp = np.cumsum(truth_sorted)
    fp = np.cumsum(1 - truth_sorted)

    tpr = np.concatenate([[0.0], tp / positives, [1.0]])
    fpr = np.concatenate([[0.0], fp / negatives, [1.0]])
    return float(np.trapezoid(tpr, fpr))


def summarize_edge_recovery(
    truth: np.ndarray,
    scores: np.ndarray,
    predicted: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = classification_metrics(truth, predicted, mask=mask)
    metrics["auprc"] = auprc(truth, scores, mask=mask)
    metrics["auroc"] = auroc(truth, scores, mask=mask)
    return metrics


def _flatten_pair(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        mask = np.ones_like(a, dtype=bool)
    return np.asarray(a)[mask].astype(bool), np.asarray(b)[mask]
