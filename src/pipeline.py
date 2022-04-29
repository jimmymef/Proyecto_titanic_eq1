from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from feature_engineering.transformers import transformers_categorical as tc
from feature_engineering.transformers import transformers_numerical as tm
from data.utils import config as cf


titanic_pipeline = Pipeline(
    [
        ("cabin_only_letter", tc.Cabin_Letter_Extractor()),
        ("drop_columns", tc.DropColumns()),
        ("missing_indicator", tm.Missing_Indicator(variables=cf.NUMERICAL_VARS)),
        (
            "categorical_imputer",
            tc.CategoricalImputerEncoder(variables=cf.CATEGORICAL_VARS),
        ),
        ("median_imputation", tm.MedianImputer(variables=cf.NUMERICAL_VARS)),
        ("rare_labels", tc.RareLabelEncoder(tol=0.05, variables=cf.CATEGORICAL_VARS)),
        ("dummy_vars", tc.OneHotEncoderImputer(variables=cf.CATEGORICAL_VARS)),
        ("scaling", MinMaxScaler()),
        (
            "log_reg",
            LogisticRegression(
                C=0.0005, class_weight="balanced", random_state=cf.SEED_MODEL
            ),
        ),
    ]
)

df = pd.read_csv(cf.URL, na_values="?")

df.to_csv("data/raw/raw_titanic.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(cf.TARGET, axis=1), df[cf.TARGET], test_size=0.2, random_state=cf.SEED_SPLIT
)

X_train.to_csv(cf.TRAIN_DATA_FILE, index=False)
X_test.to_csv(cf.TEST_DATA_FILE, index=False)

titanic_pipeline.fit_transform(X_train, y_test)

class_pred = titanic_pipeline.predict(X_test)
proba_pred = titanic_pipeline.predict_proba(X_test)[:, 1]
print("test roc-auc : {}".format(roc_auc_score(y_test, proba_pred)))
print("test accuracy: {}".format(accuracy_score(y_test, class_pred)))
