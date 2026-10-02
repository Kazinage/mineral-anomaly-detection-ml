"""Preprocessing utilities for mineralogical anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedMinerals:
    matrix: np.ndarray
    scaled: np.ndarray
    pca_scores: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    pca: PCA


def prepare_numeric_features(
    df: pd.DataFrame,
    *,
    exclude=("Name",),
    variance_threshold=0.0,
    pca_variance=0.95,
):
    """Select numeric features, remove low-variance columns, scale and apply PCA."""
    numeric = (
        df.drop(columns=[c for c in exclude if c in df], errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
    )
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    selector = VarianceThreshold(threshold=variance_threshold)
    X = selector.fit_transform(numeric)
    names = numeric.columns[selector.get_support()].tolist()

    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    pca = PCA(n_components=pca_variance, svd_solver="full").fit(Z)
    P = pca.transform(Z)

    return PreparedMinerals(X, Z, P, names, scaler, pca)
