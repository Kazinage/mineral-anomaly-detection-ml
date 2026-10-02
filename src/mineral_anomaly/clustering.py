"""Clustering of minerals flagged by either anomaly detector."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def union_anomaly_mask(isolation_flag, svm_flag):
    return np.asarray(isolation_flag, dtype=bool) | np.asarray(svm_flag, dtype=bool)


def cluster_anomalies(pca_scores, anomaly_mask, *, k=3, random_state=42):
    """Cluster only the union of detected anomalies in PCA space."""
    P = np.asarray(pca_scores, dtype=float)
    mask = np.asarray(anomaly_mask, dtype=bool)
    model = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    labels = model.fit_predict(P[mask])
    return labels, model


def elbow_inertia(pca_scores, anomaly_mask, candidates=range(1, 8), random_state=42):
    P = np.asarray(pca_scores, dtype=float)
    mask = np.asarray(anomaly_mask, dtype=bool)
    return {
        int(k): float(
            KMeans(n_clusters=k, n_init=30, random_state=random_state)
            .fit(P[mask])
            .inertia_
        )
        for k in candidates
    }
