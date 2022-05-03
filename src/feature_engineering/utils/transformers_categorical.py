import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class MissingCategoricalImputerEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame, y:pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X[self.variables] = X[self.variables].fillna("missing")
        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.02, variables: List[str] = None):
        self.tol = tol
        self.variables = variables

    def fit(self, X: pd.DataFrame, y:pd.DataFrame=None):
        self.valid_labels_dict = {}
        for var in self.variables:
            t = X[var].value_counts() / X.shape[0]
            self.valid_labels_dict[var] = t[t > self.tol].index.tolist()
            return(self)

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for var in self.variables:
            tmp = [
                col for col in X[var].unique() if col not in self.valid_labels_dict[var]
            ]
            X[var] = X[var].replace(to_replace=tmp, value=len(tmp) * ["Rare"])
        return X