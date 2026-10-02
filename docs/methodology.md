# Methodology notes

## Data preparation

The published dataset contained 3,112 minerals and a high-dimensional mixture of physical properties and chemical-composition variables.

The study standardized numeric features before PCA because distance- and variance-based algorithms are sensitive to feature scale. Very sparse or negligible-variance variables were removed during feature curation.

## PCA

PCA was used for dimensionality reduction, not as a substitute for geological feature selection.

The publication reports that:

- five PCs captured about 90% of total variance;
- six PCs captured about 95%;
- two PCs were used for visual plots;
- anomaly detection used a higher-dimensional PCA representation.

## Statistical reference

Because there was no independently labelled anomaly inventory, the study created a transparent statistical reference.

A mineral was considered anomalous if at least one of:

- Specific Gravity;
- Refractive Index;
- Molar Mass

lay more than two standard deviations from the population mean.

This reference is a benchmark, not geological ground truth.

## Complementary detectors

### Isolation Forest

Isolation Forest isolates globally unusual samples through random recursive partitioning. In the publication it behaved conservatively, producing high precision and low recall against the 2-sigma reference.

### One-Class SVM

One-Class SVM estimates a boundary around the dominant data cloud. In the publication it behaved more inclusively, recovering the full 2-sigma reference set at the cost of additional detections.

## Clustering

Only minerals flagged by at least one anomaly detector were passed to K-Means. The elbow method supported k=3.

The resulting groups were interpreted as broad mineralogical/geological associations rather than formal genetic classifications.

## Reproducibility boundary

The paper reports the methodological sequence and final metrics, but not every historical detector hyperparameter can be recovered unambiguously from the article alone. This package therefore makes operating-point parameters explicit. It does not hard-code guessed historical values as if they were documented facts.
