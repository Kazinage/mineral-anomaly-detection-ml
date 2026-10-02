import numpy as np
import pandas as pd
from mineral_anomaly.anomaly import statistical_2sigma
from mineral_anomaly.preprocessing import prepare_numeric_features

def test_preprocessing_and_reference():
    n=100
    df=pd.DataFrame({
        "Name":[f"M{i}" for i in range(n)],
        "Specific Gravity":np.r_[np.ones(n-1)*3,10],
        "Refractive Index":np.linspace(1.5,1.7,n),
        "Molar Mass":np.linspace(50,100,n),
        "Fe":np.linspace(0,1,n)
    })
    ref=statistical_2sigma(df)
    assert ref[-1]
    prepared=prepare_numeric_features(df,pca_variance=.95)
    assert prepared.pca_scores.shape[0] == n
