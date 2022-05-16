# sys and os are necessary to import config
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import config
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


X_train = pd.read_csv(config.PROCESSED_TRAIN_DATA_FILE)
y_train = pd.read_csv(config.TRAIN_TARGET_FILE).to_numpy().flatten()

logreg_model = LogisticRegression(
    C=0.0005, class_weight="balanced", random_state=config.RANDOM_SEED
)
logreg_model.fit(X_train, y_train)

joblib.dump(logreg_model, config.MODEL_FILE)

print("model training ready!")