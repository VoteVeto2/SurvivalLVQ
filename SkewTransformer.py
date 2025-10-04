import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import skew

# This class can be used to transform skewed variables
# Variables that show skew are log transformed, variables that are not are left untouched
# Z-scaling is also applied.
class SkewTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.log_cols = None
        self.min_max_sclar = None
        self.std_scaler = None

    def fit(self, X, y = None):
        self.min_max_sclar = MinMaxScaler(clip=True, feature_range=(0, 1))
        self.std_scaler = StandardScaler()
        X = self.min_max_sclar.fit_transform(X)
        X_log = np.log(1 + X)
        s_none = skew(X)
        s_log = skew(X_log)

        self.log_cols = np.abs(s_log) < np.abs(s_none)
        X[:,self.log_cols] = X_log[:,self.log_cols]

        self.std_scaler.fit(X)
        return self

    #transformation
    def transform(self, X, y = None):
        X = self.min_max_sclar.transform(X)
        X[:, self.log_cols] = np.log(1+ X[:, self.log_cols])
        return self.std_scaler.transform(X)