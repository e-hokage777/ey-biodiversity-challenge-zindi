import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    FixedThresholdClassifier,
)
from sklearn.datasets import make_classification
from utils import clean_training_data
from utils import create_pipeline
from argument_parse import TrainingArgs

# 1. Data
df = pd.read_csv("data/train-with-satellite-features.csv")
X = df.drop(["latitude", "longitude", "Occurrence Status"], axis=1)
y = df["Occurrence Status"]
X, y = clean_training_data(X, y)


def objective(trial):
    """
    Define the search space and the objective function for Optuna.
    """
    # 2. Hyperparameter Search Space
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
    }
    threshold = trial.suggest_float("threshold", 0.3, 0.7)

    # 3. Model & Cross-Validation
    pipeline = create_pipeline(
        X, y, RandomForestClassifier(**param), TrainingArgs(poly=None, train_path=None)
    )

    model = FixedThresholdClassifier(pipeline, threshold=threshold)

    # We use f1_macro as requested
    score = cross_val_score(
        model,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
        scoring="f1_macro",
    )
    f1_macro = score.mean()

    return f1_macro


if __name__ == "__main__":
    # 4. Create and Run the Study
    # 'direction' is maximize because higher F1 is better
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///tuning/individual_models.db",
        load_if_exists=True,
        study_name="random-forest-thresh",
    )
    study.optimize(objective, n_trials=100, n_jobs=3)

    # 5. Results
    print("--- Optimization Finished ---")
    print(f"Best F1 Macro Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 6. Train final model with best params
    best_model = FixedThresholdClassifier(
        RandomForestClassifier(
            **{k: v for k, v in study.best_params.items() if k != "threshold"}
        ),
        threshold=study.best_params["threshold"],
    )
    best_model.fit(X, y)
