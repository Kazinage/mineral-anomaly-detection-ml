"""Synthetic demonstration of the anomaly-detection pipeline."""

import numpy as np
import pandas as pd

from mineral_anomaly.anomaly import (
    detect_anomalies,
    evaluate_against_reference,
    statistical_2sigma,
)
from mineral_anomaly.clustering import cluster_anomalies, union_anomaly_mask
from mineral_anomaly.preprocessing import prepare_numeric_features


rng = np.random.default_rng(21)
n = 3000
df = pd.DataFrame({
    "Name": [f"Mineral_{i:04d}" for i in range(n)],
    "Mohs Hardness": rng.normal(5.0, 1.0, n),
    "Specific Gravity": rng.normal(3.0, 0.45, n),
    "Refractive Index": rng.normal(1.62, 0.12, n),
    "Molar Mass": rng.lognormal(4.5, 0.35, n),
    "Si": rng.gamma(1.5, 1.0, n),
    "Fe": rng.gamma(1.2, 0.8, n),
    "Ca": rng.gamma(1.1, 0.7, n),
})

idx = rng.choice(n, 120, replace=False)
df.loc[idx, "Specific Gravity"] += rng.normal(2.2, 0.3, len(idx))
df.loc[idx, "Molar Mass"] *= 2.0

prepared = prepare_numeric_features(df, pca_variance=0.95)
reference = statistical_2sigma(df)

# Illustrative demo operating points, not claimed as recovered historical parameters.
result = detect_anomalies(prepared.pca_scores, contamination=0.05, nu=0.20)

print(
    "Isolation Forest:",
    evaluate_against_reference(reference, result.isolation_forest, result.isolation_score),
)
print(
    "One-Class SVM:",
    evaluate_against_reference(reference, result.one_class_svm, result.svm_score),
)

union = union_anomaly_mask(result.isolation_forest, result.one_class_svm)
labels, _ = cluster_anomalies(prepared.pca_scores, union, k=3)
print("Union anomalies:", int(union.sum()))
print("Cluster counts:", dict(zip(*np.unique(labels, return_counts=True))))
