"""Detector training and evaluation."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pickle
from pathlib import Path


def train_logistic_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: str = None,
) -> LogisticRegression:
    """
    Train logistic regression detector (matches Lookback Lens).
    
    Args:
        X_train: [n_samples, n_features] feature matrix
        y_train: [n_samples] binary labels
        save_path: optional path to save model
    
    Returns:
        Trained LogisticRegression model
    """
    print(f"Training logistic regression on {len(X_train)} samples...")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Positive examples: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"Saved model to {save_path}")
    
    return clf


def evaluate_detector(
    clf: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate detector with standard metrics.
    
    Returns:
        {"auroc": float, "fpr_at_95tpr": float, "accuracy": float}
    """
    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    # AUROC
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    # FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    idx = np.where(tpr >= 0.95)[0]
    fpr_at_95tpr = fpr[idx[0]] if len(idx) > 0 else 1.0
    
    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    
    results = {
        "auroc": auroc,
        "fpr_at_95tpr": fpr_at_95tpr,
        "accuracy": accuracy,
    }
    
    print(f"\n=== Detector Evaluation ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"FPR@95%TPR: {fpr_at_95tpr:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return results


def load_detector(model_path: str) -> LogisticRegression:
    """Load saved detector."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
