import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from typing import List


class Missing_Indicator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, X: pd.DataFrame, col_names_list: List[str], y=None):
        X = pd.DataFrame(X).copy()
        for col_name in col_names_list:
            X[col_name + "_nan"] = X[col_name].isnull().astype(int)
        return X


class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        imp_median = SimpleImputer(strategy="median")
        imp_median.fit(X[self.variables])
        X[self.variables] = imp_median.transform(X[self.variables])
        return X
