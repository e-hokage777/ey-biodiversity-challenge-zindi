import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, TunedThresholdClassifierCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier

from argparse import ArgumentParser
from glob import glob
import joblib

from argument_parse import TrainingArgs

SAVE_PATH = None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_path", type=str, default="data/train-with-satellite-features.csv"
    )
    parser.add_argument(
        "--test-path", type=str, default="data/test-with-satellite-features.csv"
    )
    parser.add_argument("--poly", type=int, default=None)
    parser.add_argument("--save-pipeline", type=bool, default=True)
    return parser.parse_args()


def get_training_data(args):
    df = pd.read_csv(args.train_path)
    X = df.drop(["latitude", "longitude", "Occurrence Status"], axis=1)
    y = df["Occurrence Status"]


    return X, y


# def clean_training_data(df):
#     return df.drop_duplicates(subset=set(df.columns) - {"latitude", "longitude"})




def create_pipeline(X, y, args: TrainingArgs):
    num_features = X.select_dtypes("number").columns

    # Define preprocessing steps
    transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            # Scale numerical features
        ]
    )

    if args.poly:
        transformer.steps.append(("poly", PolynomialFeatures(degree=args.poly)))

    transformer.steps.append(("scaler", MinMaxScaler()))

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ("num", transformer, num_features),
        ],
        remainder="drop",
    )

    # Define the Logistic Regression model
    model = LogisticRegression(max_iter=10000, random_state=42)
    # model = XGBClassifier(n_estimators=64, max_depth=3, random_state=42)

    # Create the full pipeline
    pipeline = Pipeline([("preprocessing", preprocessor), ("classifier", model)])

    return pipeline


def get_save_path():
    global SAVE_PATH

    if SAVE_PATH:
        return SAVE_PATH
    save_folder = "train_logs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    versions = os.listdir(save_folder)
    version_name = f"version_{len(versions) + 1}"
    save_path = os.path.join(save_folder, version_name)

    os.mkdir(save_path)

    SAVE_PATH = save_path

    return save_path

def clean_training_data(X_train, y_train):
    duplicated_indices = X_train.duplicated()
    Xt = X_train[~duplicated_indices]
    yt = y_train[~duplicated_indices]
    return Xt, yt


def train(args: TrainingArgs):
    X, y = get_training_data(args)
    pipeline = create_pipeline(X, y, args)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y, test_size=0.3
    )
    X_train, y_train = clean_training_data(X_train, y_train)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    pipeline = TunedThresholdClassifierCV(pipeline, cv="prefit", refit=False, scoring="f1_macro").fit(X_test, y_test)

    ## finding best threshold

    ## computing f1_score
    # scores = cross_val_score(
    #     pipeline, X, y, cv=StratifiedKFold(n_splits=3, random_state=42, shuffle=True), scoring="f1_macro", n_jobs=-1
    # )

    # print("f1_score",np.mean(scores))

    evaluations(X_train, y_train, X_test, y_test, pipeline)
    return pipeline


def predict(pipeline, args):
    df = pd.read_csv(args.test_path)
    df = df.drop(["latitude", "longitude"], axis=1)
    return pipeline.predict(df)


def evaluations(X_train, y_train, X_test, y_test, pipeline, save_folder="train_logs"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes = axes.flatten()
    ConfusionMatrixDisplay.from_estimator(pipeline, X_train, y_train, ax=axes[0], normalize="true")
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[1], normalize="true")

    ## getting save path
    save_path = get_save_path()

    ## saving figure
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()

    ## saving classification reports
    train_classification_report = classification_report(
        y_train, pipeline.predict(X_train), output_dict=True
    )
    test_classification_report = classification_report(
        y_test, pipeline.predict(X_test), output_dict=True
    )

    train_report_df = format_report_df_columns(
        pd.DataFrame(train_classification_report), "train"
    )
    test_report_df = format_report_df_columns(
        pd.DataFrame(test_classification_report), "test"
    )

    train_report_df.to_csv(f"{save_path}/train_classification_report.csv")
    test_report_df.to_csv(f"{save_path}/test_classification_report.csv")

    print("Reports generated")


def format_report_df_columns(df, prefix):
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df


if __name__ == "__main__":
    args = parse_args()
    pipeline = train(TrainingArgs(args))

    if args.save_pipeline:
        save_path = get_save_path()
        joblib.dump(pipeline, os.path.join(save_path, "model.joblib"))
