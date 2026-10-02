"""Isolation Forest, One-Class SVM and statistical anomaly benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.svm import OneClassSVM


@dataclass
class AnomalyResult:
    isolation_forest: np.ndarray
    one_class_svm: np.ndarray
    isolation_score: np.ndarray
    svm_score: np.ndarray


def statistical_2sigma(
    df,
    columns=("Specific Gravity", "Refractive Index", "Molar Mass"),
):
    """Reference anomaly: outside mean +/- 2 sigma in at least one key property."""
    mask = np.zeros(len(df), dtype=bool)
    for col in columns:
        x = np.asarray(df[col], dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        mask |= np.abs(x - mu) > 2.0 * sd
    return mask


def detect_anomalies(X, *, contamination, nu, gamma="scale", random_state=42):
    """Fit complementary unsupervised detectors.

    Operating parameters are explicit so that a new analysis documents its
    assumptions rather than inheriting hidden defaults.
    """
    X = np.asarray(X, dtype=float)

    iso = IsolationForest(
        contamination=float(contamination),
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    ).fit(X)
    iso_flag = iso.predict(X) == -1
    iso_score = -iso.decision_function(X)

    svm = OneClassSVM(nu=float(nu), kernel="rbf", gamma=gamma).fit(X)
    svm_flag = svm.predict(X) == -1
    svm_score = -svm.decision_function(X)

    return AnomalyResult(iso_flag, svm_flag, iso_score, svm_score)


def evaluate_against_reference(reference, predicted, score):
    y = np.asarray(reference, dtype=int)
    p = np.asarray(predicted, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, p, average="binary", zero_division=0
    )
    auc = roc_auc_score(y, np.asarray(score, dtype=float))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auc),
    }
