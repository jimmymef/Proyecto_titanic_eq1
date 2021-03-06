URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
DROP_COLS = ["boat", "body", "home.dest", "ticket", "name"]
TARGET = "survived"
RANDOM_SEED = 404
NUMERICAL_VARS = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_VARS = ["sex", "cabin", "embarked", "title"]
TRAIN_DATA_FILE = "data/cleaned/train.csv"
TEST_DATA_FILE = "data/cleaned/test.csv"
TRAIN_TARGET_FILE = "data/cleaned/y_train.csv"
TEST_TARGET_FILE = "data/cleaned/y_test.csv"
PROCESSED_TRAIN_DATA_FILE = "data/processed/train.csv"
PROCESSED_TEST_DATA_FILE = "data/processed/test.csv"
MODEL_FILE = "models/logistic_regression/log_reg.pkl"