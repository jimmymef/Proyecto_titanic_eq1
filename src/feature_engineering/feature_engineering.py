import pandas as pd
import numpy as np
from utils import transformers_categorical as tc
from utils import transformers_numerical as tn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from utils import config

train_df = pd.read_csv("data/cleaned/train.csv")
test_df = pd.read_csv("data/cleaned/test.csv")

# Categorical variables pipeline + missing numerical indicator
numeric_missing_indicator_ = tn.Missing_Indicator(config.NUMERICAL_VARS)
missing_imputer = tc.MissingCategoricalImputerEncoder(variables=config.CATEGORICAL_VARS)
rare_label_encoder = tc.RareLabelEncoder(variables=["cabin"])

categorical_pipe = Pipeline(
    [
        ("numeric_mi", numeric_missing_indicator_),
        ("categorical_mi", missing_imputer),
        ("rare_label_enc", rare_label_encoder),
    ]
)
categorical_pipe.fit(train_df)

processed_train_df = categorical_pipe.transform(train_df)
processed_test_df = categorical_pipe.transform(test_df)

## One hot encoding (using pandas).
processed_train_df = pd.get_dummies(
    processed_train_df, columns=config.CATEGORICAL_VARS, drop_first=True
)
processed_test_df = pd.get_dummies(
    processed_test_df, columns=config.CATEGORICAL_VARS, drop_first=True
)

missing_onehot_cols = list(
    set(processed_train_df.columns) - set(processed_test_df.columns)
)
for colname in missing_onehot_cols:
    processed_test_df[colname] = 0

# Numerical variables (Could not be done with a pipeline)

numerical_median_imputer = SimpleImputer(strategy="median")
min_max_scaler = MinMaxScaler()

numeric_pipeline = Pipeline(
    [("median_imputer", numerical_median_imputer), ("min_max_scaler", min_max_scaler)]
)
numeric_col_transformer = ColumnTransformer(
    [("numeric_transforms", numeric_pipeline, config.NUMERICAL_VARS)]
)

processed_train_df[config.NUMERICAL_VARS] = numeric_col_transformer.fit_transform(
    processed_train_df
)
processed_test_df[config.NUMERICAL_VARS] = numeric_col_transformer.fit_transform(
    processed_test_df
)

processed_test_df = processed_test_df[processed_train_df.columns]

# Save processed DFs
processed_train_df.to_csv(config.PROCESSED_TRAIN_DATA_FILE, index=False)
processed_test_df.to_csv(config.PROCESSED_TEST_DATA_FILE, index=False)
