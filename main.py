from src.data.utils import config as cf
import pandas as pd
import re
from src.feature_engineering.transformers import transformers_categorical as tc
from src.feature_engineering.transformers import transformers_numerical as tn
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(cf.URL, na_values="?")
df["cabin"] = df.cabin.apply(lambda x: x[0] if type(x) == str else x)
df["title"] = df.name.apply(lambda x: x.split(",")[1].split(".")[0])
df.drop(columns=cf.DROP_COLS, inplace=True)
print(df)

df2 = df
rare_labels = tn.MedianImputer(variables=cf.NUMERICAL_VARS)
rare_labels.fit(df2)
df2 = rare_labels.transform(df2)

print(df2)
