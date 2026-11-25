import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from src.config import MODEL_PATH, METRICS_PATH, RANDOM_STATE, CV_FOLDS, FEATURE_NAMES

class WineQualityModelPipeline:
    def __init__(self, n_estimators: int = 150, max_depth: int = 12):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",  # Addresses the 13.5% minority class imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        self.metrics: Dict[str, Any] = {}

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform 5-fold Stratified cross-validation with multiple metrics."""
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        cv_results = cross_validate(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_summary = {
            "cv_accuracy_mean": float(np.mean(cv_results['test_accuracy'])),
            "cv_accuracy_std": float(np.std(cv_results['test_accuracy'])),
            "cv_precision_mean": float(np.mean(cv_results['test_precision'])),
            "cv_recall_mean": float(np.mean(cv_results['test_recall'])),
            "cv_f1_mean": float(np.mean(cv_results['test_f1'])),
            "cv_roc_auc_mean": float(np.mean(cv_results['test_roc_auc'])),
        }
        return cv_summary

    def train_and_evaluate(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Train the classifier and compute comprehensive test set evaluation metrics."""
        # Fit on training data
        self.model.fit(X_train, y_train)

        # Test set predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred).tolist()

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": cm,
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "feature_importance": {
                name: round(float(imp), 4)
                for name, imp in zip(FEATURE_NAMES, self.model.feature_importances_)
            }
        }
        return self.metrics

    def save(self, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH):
        """Serialize model and metadata."""
        joblib.dump(self.model, model_path)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH):
        """Load serialized model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        instance = cls()
        instance.model = joblib.load(model_path)
        return instance

    def predict_one(self, feature_dict: Dict[str, float]) -> Tuple[int, float, float]:
        """Inference for a single record. Returns (class, prob_bad, prob_good)."""
        df = pd.DataFrame([feature_dict])[FEATURE_NAMES]
        proba = self.model.predict_proba(df)[0]
        pred_class = int(np.argmax(proba))
        return pred_class, float(proba[0]), float(proba[1])
