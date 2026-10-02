# Mineral Anomaly Detection with Machine Learning

**Unsupervised detection and clustering of rare/anomalous minerals from multivariate mineralogical properties.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.acags.2025.100250-blue)](https://doi.org/10.1016/j.acags.2025.100250)

Reference implementation associated with:

> **Sharapatov, A.; Saduov, A.; Assirbek, N.; Abdyrov, M.; Zhumabayev, B. (2025).**  
> *Prediction of rare and anomalous minerals using anomaly detection and machine learning techniques.*  
> **Applied Computing and Geosciences, 26, 100250.**  
> https://doi.org/10.1016/j.acags.2025.100250

Alisher Saduov is the corresponding author of the published article.

## Research objective

The project asks whether high-dimensional physical and chemical mineral properties can be used to identify unusual minerals without a conventional labelled anomaly dataset.

The workflow combines complementary unsupervised methods:

- StandardScaler for comparable feature scales;
- PCA for dimensionality reduction;
- Isolation Forest for conservative global outliers;
- One-Class SVM for more inclusive boundary-based anomaly detection;
- a transparent **2σ statistical reference** for model evaluation;
- K-Means to characterize structure within the union of detected anomalies.

## Dataset described in the article

The published study used **3,112 mineral entries** assembled from public mineralogical sources. The dataset contained approximately **140 physical/chemical features**, including elemental composition and properties such as:

- crystal structure;
- Mohs hardness;
- specific gravity;
- refractive index;
- molar mass and molar volume;
- major and trace elemental composition.

The article reports that five principal components retained more than 90% of the variance, while two PCs were used for visualization.

## Workflow

```text
Public mineralogical database
          |
          v
Cleaning + numeric feature curation
          |
          v
StandardScaler
          |
          v
PCA (5-6 PCs for modelling; 2 PCs for visualization)
          |
      +---+-------------------+
      |                       |
      v                       v
Isolation Forest        One-Class SVM
      |                       |
      +-----------+-----------+
                  |
                  v
       2σ statistical benchmark
                  |
                  v
    Union of detected anomalies
                  |
                  v
       K-Means on anomaly subset
                  |
                  v
       k = 3 geological groups
```

## Published results

The statistical reference classified **564 minerals** as anomalous when at least one of Specific Gravity, Refractive Index or Molar Mass exceeded ±2 standard deviations from the population mean.

The study reported:

| Model | Detected anomalies | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | 156 | 1.000 | 0.277 | 0.433 | 0.638 |
| One-Class SVM | 663 | 0.851 | 1.000 | 0.919 | 0.981 |

The two methods therefore behaved differently: Isolation Forest was conservative, while One-Class SVM was more inclusive. K-Means was then applied to the union of detected anomalies, with the elbow method supporting **k = 3**.

The article interpreted the three groups broadly as mineral assemblages associated with evaporitic/alkaline, metamorphic-metasomatic/hydrothermal, and magmatic plus secondary weathering environments.

## Repository structure

```text
.
├── data/README.md
├── docs/methodology.md
├── examples/synthetic_demo.py
├── src/mineral_anomaly/
│   ├── __init__.py
│   ├── anomaly.py
│   ├── clustering.py
│   └── preprocessing.py
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/Kazinage/mineral-anomaly-detection-ml.git
cd mineral-anomaly-detection-ml
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python examples/synthetic_demo.py
```

## Parameter transparency

The article documents the methodological sequence and reported operating points, but not every historical software parameter is recoverable from the publication alone. For that reason, this implementation makes detector operating parameters such as Isolation Forest `contamination` and One-Class SVM `nu` explicit inputs rather than silently presenting guessed defaults as the original settings.

The synthetic demo uses illustrative values only.

## Data and article licensing

The article is open access under **CC BY-NC-ND 4.0**. This repository does not copy the article text or redistribute the article PDF. The software implementation is independently released under MIT.

Public source data should be obtained from their original providers and used according to the applicable terms.

## Author

**Alisher Saduov, PhD**  
Geophysics | GeoAI | Mineral Exploration | Machine Learning  
Satbayev University, Kazakhstan  
ORCID: 0000-0003-1501-7772

## License

Code is MIT licensed. External datasets and publication content retain their original licenses.
