import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from yellowbrick.target import ClassBalance
from yellowbrick.classifier import (
    ROCAUC,
    PrecisionRecallCurve,
    ClassificationReport,
    ClassPredictionError,
    DiscriminationThreshold,
    ConfusionMatrix,
)


import dagshub
dagshub.init(repo_owner='AnandaNekyP', repo_name='Membangun_model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AnandaNekyP/Membangun_model.mlflow")

# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.autolog()

INPUT_PATH = "recruitment_data_preprocessing.csv"


def load_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df):
    cb = ClassBalance(labels=["Not Hired", "Hired"])
    cb.fit(df["HiringDecision"])
    cb.show(outpath="viz/Class Balance.png")

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


def evaluate_model(model, x_train, y_train, x_test, y_test):
    classes = ["Not Hired", "Hired"]

    visualizers = {
        "Classification Report": ClassificationReport(
            model, classes=classes, support=True
        ),
        "Confusion Matrix": ConfusionMatrix(model, classes=classes),
        "ROC AUC": ROCAUC(model, classes=classes),
        "Precision Recall Curve": PrecisionRecallCurve(model, classes=classes),
        "Class Prediction Error": ClassPredictionError(model, classes=classes),
        # "Discrimination Threshold": DiscriminationThreshold(model, classes=classes),
    }
    for name, viz in visualizers.items():
        viz.fit(x_train, y_train)
        viz.score(x_test, y_test)
        viz.show(outpath=f"viz/{name}.png", clear_figure=True)
        plt.close('all')



with mlflow.start_run():
    df = load_data(INPUT_PATH)
    x_train, y_train, x_test, y_test = split_data(df)
    model = train_model(x_train, y_train)
    evaluate_model(model, x_train, y_train, x_test, y_test)
