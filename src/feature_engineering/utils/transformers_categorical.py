import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class MissingCategoricalImputerEncoder(BaseEstimator, TransformerMixin):
    """A transformer that fills missing value with the legend 'missing'"""

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Method fit is required by the parent class, however it does nothing.

        Parameters
        ----------
        X: pd.DataFrame : A dataframe with missing values (pd.NA)

        y: pd.DataFrame : Required by the parent class
             (Default value = None)

        Returns
        -------
        Nothing
        """
        return self

    def transform(self, X: pd.DataFrame):
        """This method will fill the missing values of the variables defined in
        __init__  with the word 'missing'

        Parameters
        ----------
        X: pd.DataFrame : The dataframe to transform.


        Returns
        -------
        A pd.Dataframe with the values 'missing' which previously where pd.NA
        """
        X = X.copy()
        X[self.variables] = X[self.variables].fillna("missing")
        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """Rare label encoder will add the label rare to categorical variables if
    the original label is found less times than those defined by tol"""

    def __init__(self, tol=0.02, variables: List[str] = None):
        self.tol = tol
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Method fit is required by the parent class, however it does nothing.

        Parameters
        ----------
        X: pd.DataFrame :

        y: pd.DataFrame : Required by the parent class. Don't add a dataframe here.
             (Default value = None)

        Returns
        -------
        Nothing
        """
        self.valid_labels_dict = {}
        for var in self.variables:
            t = X[var].value_counts() / X.shape[0]
            self.valid_labels_dict[var] = t[t > self.tol].index.tolist()
            return self

    def transform(self, X: pd.DataFrame):
        """This method will convert the original label of the categorical
        variables selected in __init__ to 'Rare'  if the original label is found
        less times than those defined by tol in init

        Parameters
        ----------
        X: pd.DataFrame : The dataframe to transform.


        Returns
        -------
        A copy of X with the replaced labels in the selected variables.
        """
        X = X.copy()
        for var in self.variables:
            tmp = [
                col for col in X[var].unique() if col not in self.valid_labels_dict[var]
            ]
            X[var] = X[var].replace(to_replace=tmp, value=len(tmp) * ["Rare"])
        return X
