import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class Missing_Indicator(BaseEstimator, TransformerMixin):
    """This transformer will add a new column for each column selected in
    self.variables to a pd.Dataframe indicating with one if the value is missing
    and 0 if value is not missing"""

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Method fit is required by the parent class, however it does nothing.

        Parameters
        ----------
        X: pd.DataFrame : A dataframe with numeric missing values (pd.NA)

        y: pd.DataFrame :
             (Default value = None)

        Returns
        -------
        Nothing
        """
        return self

    def transform(self, X: pd.DataFrame):
        """Transform will add a new column for each column selected in
        self.variables to a pd.Dataframe indicating with one if the value is missing
        and 0 if value is not missing.

            Parameters
            ----------
            X: pd.DataFrame :


            Returns
            -------
            A transformed dataframe with a new column for each column selected in
        self.variables
        """
        X = X.copy()
        for var in self.variables:
            X[var + "_nan"] = X[var].isnull().astype(int)
        return X
