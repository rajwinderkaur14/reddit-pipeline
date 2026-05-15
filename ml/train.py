# ml/train.py

import os
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
DELTA_PATH   = "data/processed/hn_stories_delta"
MLFLOW_DIR   = "mlruns"
EXPERIMENT   = "hn-virality-prediction"

# Features we'll use to train the model
# These are all columns we engineered in Phase 3
FEATURES = [
    "score",
    "comments",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "title_length",
    "title_word_count",
    "has_url",
    "is_show_hn",
    "is_ask_hn",
    "comment_score_ratio",
]
TARGET = "is_viral"


# ── 1. Load data from Delta Lake ───────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Read parquet files from Delta Lake into a pandas DataFrame.
    We use pandas here (not Spark) because the dataset is small enough
    and sklearn/xgboost work with pandas natively.
    """
    import glob
    parquet_files = glob.glob(f"{path}/**/*.parquet", recursive=True)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at {path}. Run Spark job first!")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df  = pd.concat(dfs, ignore_index=True)
    df  = df.drop_duplicates(subset=["id"])

    print(f"Loaded {len(df)} rows from Delta Lake")
    print(f"Columns: {list(df.columns)}")
    print(f"Viral distribution:\n{df[TARGET].value_counts()}\n")
    return df


# ── 2. Prepare features ────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame):
    """
    Select feature columns and target, handle any remaining nulls.
    Split into train (80%) and test (20%) sets.
    """
    # But wait — score IS what defines is_viral (score >= 100).
    # If we include score as a feature, the model will cheat!
    # Remove it so the model learns from everything EXCEPT the score itself.
    features_no_leakage = [f for f in FEATURES if f not in ("score", "comment_score_ratio", "comments")]

    print(f"Training features: {features_no_leakage}")

    X = df[features_no_leakage].fillna(0)
    y = df[TARGET].fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y    # ensure same viral % in train and test
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Train viral %: {y_train.mean()*100:.1f}% | Test viral %: {y_test.mean()*100:.1f}%\n")
    return X_train, X_test, y_train, y_test, features_no_leakage


# ── 3. Train model ─────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    """
    Train an XGBoost classifier.
    XGBoost = Extreme Gradient Boosting — builds many small decision trees
    where each tree corrects the mistakes of the previous one.
    It's the most common algorithm in Kaggle competitions and real DE projects.
    """
    params = {
        "n_estimators":   100,    # number of trees
        "max_depth":      4,      # how deep each tree grows
        "learning_rate":  0.1,    # how much each tree contributes
        "subsample":      0.8,    # use 80% of data per tree (prevents overfitting)
        "random_state":   42,
        "eval_metric":    "logloss",
        "use_label_encoder": False,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


# ── 4. Evaluate model ──────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """
    Metrics explained:
    - Accuracy:  % of all predictions correct
    - Precision: of stories we predicted viral, how many actually were?
    - Recall:    of actually viral stories, how many did we catch?
    - F1:        balance between precision and recall
    - ROC-AUC:   how well model separates viral from non-viral (1.0 = perfect)
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_pred_prob),
    }

    print("="*50)
    print("        MODEL EVALUATION")
    print("="*50)
    for name, val in metrics.items():
        print(f"  {name:<12}: {val:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Viral", "Viral"]))

    return metrics, y_pred, y_pred_prob


# ── 5. Plot feature importance ─────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, save_path: str):
    """Shows which features the model found most useful."""
    importance = model.feature_importances_
    indices    = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance — HN Virality Predictor", fontsize=14)
    plt.bar(range(len(feature_names)),
            importance[indices],
            color="steelblue", edgecolor="white")
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in indices],
               rotation=45, ha="right")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


# ── 6. Plot confusion matrix ───────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, save_path: str):
    """
    Confusion matrix shows:
    - True Positives:  correctly predicted viral
    - True Negatives:  correctly predicted not viral
    - False Positives: predicted viral but wasn't
    - False Negatives: predicted not viral but was
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Viral", "Viral"])
    ax.set_yticklabels(["Not Viral", "Viral"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("ml/plots", exist_ok=True)
    os.makedirs("ml/models", exist_ok=True)

    # Set up MLflow — tracks all experiments locally in mlruns/
    mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_DIR)}")
    mlflow.set_experiment(EXPERIMENT)

    print("Starting ML training pipeline...")
    print("="*50)

    # Load and prepare data
    df = load_data(DELTA_PATH)
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df)

    # Start MLflow run — everything inside is tracked automatically
    with mlflow.start_run(run_name="xgboost-baseline"):

        # Train
        print("Training XGBoost model...")
        model, params = train_model(X_train, y_train)

        # Evaluate
        metrics, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)

        # Generate plots
        plot_feature_importance(model, feature_names, "ml/plots/feature_importance.png")
        plot_confusion_matrix(y_test, y_pred, "ml/plots/confusion_matrix.png")

        # Log everything to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("ml/plots/feature_importance.png")
        mlflow.log_artifact("ml/plots/confusion_matrix.png")

        # Save the model to MLflow registry
        mlflow.xgboost.log_model(model, "model")

        # Save model locally too
        model.save_model("ml/models/xgboost_virality.json")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print(f"Model saved to ml/models/xgboost_virality.json")

    print("\n" + "="*50)
    print("Training complete!")
    print("Run: mlflow ui --backend-store-uri mlruns")
    print("Then open: http://localhost:5000")
    print("="*50)
