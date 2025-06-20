import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

mlflow.autolog()

INPUT_PATH = "recruitment_data_preprocessing.csv"


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = mlflow.data.from_pandas(train_df, name="train")
    x_train = train_dataset.df.drop(["HiringDecision"], axis=1)
    y_train = train_dataset.df["HiringDecision"]

    test_dataset = mlflow.data.from_pandas(test_df, name="test")
    x_test = test_dataset.df.drop(["HiringDecision"], axis=1)
    y_test = test_dataset.df["HiringDecision"]
    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train, y_train)
    return model


with mlflow.start_run():
    df = load_data(INPUT_PATH)
    x_train, y_train, x_test, y_test = split_data(df)
    model = train_model(x_train, y_train)
