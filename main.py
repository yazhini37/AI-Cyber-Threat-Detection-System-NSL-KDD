import pandas as pd
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    """
    Load NSL-KDD data from a txt file.
    The files do not have headers, so we add them manually.
    """
    columns = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
        "difficulty",
    ]

    # NSL-KDD files are comma separated
    return pd.read_csv(path, header=None, names=columns)


def encode_label(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Convert label column into numeric values.
    Here we do binary classification:
    - 0 for normal
    - 1 for any kind of attack
    """
    train_df["label"] = (train_df["label"] != "normal").astype(int)
    test_df["label"] = (test_df["label"] != "normal").astype(int)
    return train_df, test_df


def main():
    # 1. Load train and test data
    train_path = "KDDTrain+.txt"
    test_path = "KDDTest+.txt"

    train_df = load_data(train_path)
    test_df = load_data(test_path)

    # 2. Convert label column to numeric (0 = normal, 1 = attack)
    train_df, test_df = encode_label(train_df, test_df)

    # 3. Split features (X) and labels (y)
    feature_cols = [col for col in train_df.columns if col not in ["label", "difficulty"]]
    X = train_df[feature_cols]
    y = train_df["label"]

    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    # 4. Build a clean preprocessing + model pipeline
    # - OneHotEncoder safely handles categorical columns (and unknown categories in test set)
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # 5. Cross-validation on training set (more reliable than a single split)
    # We create out-of-fold probabilities for EACH training sample.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        fold_model = clone(pipeline)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]

    # 6. Pick a threshold using cross-validated (out-of-fold) probabilities
    candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    best_threshold = 0.50
    best_f1 = -1.0

    for t in candidates:
        oof_pred = (oof_proba >= t).astype(int)
        f1 = f1_score(y, oof_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Cross-validation (OOF) report: realistic estimate on training distribution
    y_oof_pred = (oof_proba >= best_threshold).astype(int)
    oof_acc = accuracy_score(y, y_oof_pred)
    oof_precision = precision_score(y, y_oof_pred, zero_division=0)
    oof_recall = recall_score(y, y_oof_pred, zero_division=0)
    oof_f1 = f1_score(y, y_oof_pred, zero_division=0)

    print(f"Best threshold (chosen by 5-fold CV): {best_threshold:.2f} (CV F1={best_f1:.4f})")
    print(f"CV Accuracy: {oof_acc:.4f}")
    print(f"CV Precision (attack=1): {oof_precision:.4f}")
    print(f"CV Recall (attack=1):    {oof_recall:.4f}")
    print(f"CV F1-score (attack=1):  {oof_f1:.4f}")
    print("CV Confusion Matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y, y_oof_pred))

    # 7. Train final model on FULL training data
    pipeline.fit(X, y)

    # 8. Evaluate on the test set (final report)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Test Accuracy: {acc:.4f}")

    # Confusion matrix to see detailed counts
    # Rows: true labels, Columns: predicted labels
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows = true, cols = predicted):")
    print(cm)

    print(f"Precision (attack=1): {precision:.4f}")
    print(f"Recall (attack=1):    {recall:.4f}")
    print(f"F1-score (attack=1):  {f1:.4f}")

    # Extra: full report (good for final-year / resume screenshots)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"], zero_division=0))

    # 9. Save the trained pipeline (preprocess + model) for reuse
    joblib.dump(pipeline, "nsl_kdd_rf_pipeline.joblib")
    print("Saved model to: nsl_kdd_rf_pipeline.joblib")

    # 10. Show top feature importances (helps for reports/interviews)
    # We extract the final feature names after OneHotEncoding.
    preprocess = pipeline.named_steps["preprocess"]
    rf = pipeline.named_steps["model"]

    cat_encoder = preprocess.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = cat_feature_names + numeric_cols

    importances = rf.feature_importances_
    top_n = 15
    top_idx = importances.argsort()[::-1][:top_n]

    print(f"\nTop {top_n} Feature Importances:")
    for rank, i in enumerate(top_idx, start=1):
        print(f"{rank:>2}. {all_feature_names[i]}: {importances[i]:.6f}")


if __name__ == "__main__":
    main()