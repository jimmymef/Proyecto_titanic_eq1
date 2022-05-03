import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class Missing_Indicator(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for var in self.variables:
            X[var + "_nan"] = X[var].isnull().astype(int)
        return X
