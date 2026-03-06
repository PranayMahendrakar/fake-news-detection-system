"""
ML Models for Fake News Detection
Implements Logistic Regression, Naive Bayes, and SVM classifiers
with TF-IDF vectorization and cross-validation
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: Any = None
    cv_scores: List[float] = field(default_factory=list)
    classification_report: str = ""

    @property
    def cv_mean(self) -> float:
        return np.mean(self.cv_scores) if self.cv_scores else 0.0

    @property
    def cv_std(self) -> float:
        return np.std(self.cv_scores) if self.cv_scores else 0.0

    def to_dict(self) -> Dict:
        return {
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'roc_auc': round(self.roc_auc, 4),
            'cv_mean': round(self.cv_mean, 4),
            'cv_std': round(self.cv_std, 4),
        }


class FakeNewsClassifier:
    """
    Fake News Detection using multiple ML models.
    
    Models:
    - Logistic Regression (with L2 regularization)
    - Naive Bayes (Complement NB for imbalanced data)
    - SVM (Linear SVC with probability calibration)
    
    Features TF-IDF vectorization with n-gram support.
    """

    MODEL_CONFIGS = {
        'logistic_regression': {
            'vectorizer': TfidfVectorizer(
                max_features=100000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            ),
            'classifier': LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs',
                random_state=42,
                class_weight='balanced'
            ),
            'display_name': 'Logistic Regression',
            'description': 'Fast, interpretable linear classifier with L2 regularization'
        },
        'naive_bayes': {
            'vectorizer': TfidfVectorizer(
                max_features=80000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=False
            ),
            'classifier': ComplementNB(alpha=0.1),
            'display_name': 'Naive Bayes',
            'description': 'Complement NB — excellent for imbalanced text classification'
        },
        'svm': {
            'vectorizer': TfidfVectorizer(
                max_features=100000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            ),
            'classifier': CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight='balanced'),
                cv=3
            ),
            'display_name': 'Support Vector Machine',
            'description': 'Linear SVM with probability calibration — high accuracy on text'
        }
    }

    LABEL_MAP = {0: 'FAKE', 1: 'REAL'}
    REVERSE_LABEL_MAP = {'FAKE': 0, 'REAL': 1}

    def __init__(self):
        self.models: Dict[str, Pipeline] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.is_trained: Dict[str, bool] = {name: False for name in self.MODEL_CONFIGS}
        self.feature_names: Optional[List[str]] = None

    def _build_pipeline(self, model_name: str) -> Pipeline:
        """Build sklearn Pipeline for a model."""
        config = self.MODEL_CONFIGS[model_name]
        return Pipeline([
            ('vectorizer', config['vectorizer']),
            ('classifier', config['classifier'])
        ])

    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        model_names: Optional[List[str]] = None,
        cv_folds: int = 5
    ) -> Dict[str, ModelMetrics]:
        """
        Train specified models (or all if none specified).
        
        Args:
            X_train: Preprocessed text samples
            y_train: Binary labels (0=FAKE, 1=REAL)
            model_names: List of models to train (default: all)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of training metrics per model
        """
        if model_names is None:
            model_names = list(self.MODEL_CONFIGS.keys())

        train_metrics = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name in model_names:
            logger.info(f"Training {name}...")
            try:
                pipeline = self._build_pipeline(name)

                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=cv, scoring='f1', n_jobs=-1
                )

                # Train on full training set
                pipeline.fit(X_train, y_train)
                self.models[name] = pipeline
                self.is_trained[name] = True

                # Store feature names from vectorizer
                if self.feature_names is None:
                    self.feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()

                # Training set metrics
                y_pred = pipeline.predict(X_train)
                metrics = ModelMetrics(
                    accuracy=accuracy_score(y_train, y_pred),
                    precision=precision_score(y_train, y_pred, zero_division=0),
                    recall=recall_score(y_train, y_pred, zero_division=0),
                    f1_score=f1_score(y_train, y_pred, zero_division=0),
                    cv_scores=cv_scores.tolist(),
                    classification_report=classification_report(
                        y_train, y_pred,
                        target_names=['FAKE', 'REAL']
                    )
                )

                # ROC-AUC if probability available
                if hasattr(pipeline, 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_train)[:, 1]
                    metrics.roc_auc = roc_auc_score(y_train, y_proba)

                self.metrics[name] = metrics
                train_metrics[name] = metrics
                logger.info(f"  {name} CV F1: {metrics.cv_mean:.4f} ± {metrics.cv_std:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                raise

        return train_metrics

    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray,
        model_name: Optional[str] = None
    ) -> Dict[str, ModelMetrics]:
        """
        Evaluate trained models on test data.
        
        Returns test set metrics for specified or all models.
        """
        models_to_eval = [model_name] if model_name else list(self.models.keys())
        eval_metrics = {}

        for name in models_to_eval:
            if not self.is_trained.get(name, False):
                logger.warning(f"Model {name} not trained, skipping.")
                continue

            pipeline = self.models[name]
            y_pred = pipeline.predict(X_test)

            metrics = ModelMetrics(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1_score=f1_score(y_test, y_pred, zero_division=0),
                confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
                classification_report=classification_report(
                    y_test, y_pred,
                    target_names=['FAKE', 'REAL']
                )
            )

            if hasattr(pipeline, 'predict_proba'):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics.roc_auc = roc_auc_score(y_test, y_proba)

            eval_metrics[name] = metrics
            logger.info(
                f"{name} Test — Acc: {metrics.accuracy:.4f}, "
                f"F1: {metrics.f1_score:.4f}, AUC: {metrics.roc_auc:.4f}"
            )

        return eval_metrics

    def predict(
        self,
        text: str,
        model_name: str = 'logistic_regression'
    ) -> Dict[str, Any]:
        """
        Predict single text classification with confidence score.
        
        Returns:
            {
                'label': 'FAKE' | 'REAL',
                'confidence': float (0-1),
                'probabilities': {'FAKE': float, 'REAL': float},
                'model': str
            }
        """
        if not self.is_trained.get(model_name, False):
            raise ValueError(f"Model '{model_name}' is not trained.")

        pipeline = self.models[model_name]
        prediction = pipeline.predict([text])[0]
        label = self.LABEL_MAP[prediction]

        result = {
            'label': label,
            'prediction': int(prediction),
            'model': self.MODEL_CONFIGS[model_name]['display_name'],
            'model_key': model_name,
        }

        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba([text])[0]
            result['probabilities'] = {
                'FAKE': round(float(proba[0]), 4),
                'REAL': round(float(proba[1]), 4)
            }
            result['confidence'] = round(float(max(proba)), 4)
        else:
            result['confidence'] = 1.0
            result['probabilities'] = {'FAKE': float(1 - prediction), 'REAL': float(prediction)}

        return result

    def predict_all_models(self, text: str) -> Dict[str, Dict]:
        """Run prediction through all trained models."""
        results = {}
        for name in self.models:
            if self.is_trained.get(name, False):
                results[name] = self.predict(text, name)
        return results

    def get_top_features(
        self,
        model_name: str,
        n_features: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features (words) for FAKE and REAL classes.
        Used for explainable AI.
        
        Returns dict with 'fake_features' and 'real_features'.
        """
        if not self.is_trained.get(model_name, False):
            raise ValueError(f"Model '{model_name}' is not trained.")

        pipeline = self.models[model_name]
        vectorizer = pipeline.named_steps['vectorizer']
        classifier = pipeline.named_steps['classifier']
        feature_names = vectorizer.get_feature_names_out()

        result = {'fake_features': [], 'real_features': []}

        # Get coefficients/feature importances
        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_
            if coef.ndim > 1:
                coef = coef[0]

            # Top features for REAL (positive coef)
            top_real_idx = np.argsort(coef)[-n_features:][::-1]
            result['real_features'] = [
                (feature_names[i], round(float(coef[i]), 4))
                for i in top_real_idx
            ]

            # Top features for FAKE (negative coef)
            top_fake_idx = np.argsort(coef)[:n_features]
            result['fake_features'] = [
                (feature_names[i], round(float(abs(coef[i])), 4))
                for i in top_fake_idx
            ]

        elif hasattr(classifier, 'estimator') and hasattr(classifier.estimator, 'coef_'):
            # CalibratedClassifierCV wraps LinearSVC
            coef = classifier.estimator.coef_[0]
            top_real_idx = np.argsort(coef)[-n_features:][::-1]
            result['real_features'] = [
                (feature_names[i], round(float(coef[i]), 4))
                for i in top_real_idx
            ]
            top_fake_idx = np.argsort(coef)[:n_features]
            result['fake_features'] = [
                (feature_names[i], round(float(abs(coef[i])), 4))
                for i in top_fake_idx
            ]

        elif hasattr(classifier, 'feature_log_prob_'):
            # Naive Bayes
            log_probs = classifier.feature_log_prob_
            fake_log_probs = log_probs[0]
            real_log_probs = log_probs[1]
            diff = real_log_probs - fake_log_probs

            top_real_idx = np.argsort(diff)[-n_features:][::-1]
            result['real_features'] = [
                (feature_names[i], round(float(diff[i]), 4))
                for i in top_real_idx
            ]

            top_fake_idx = np.argsort(diff)[:n_features]
            result['fake_features'] = [
                (feature_names[i], round(float(abs(diff[i])), 4))
                for i in top_fake_idx
            ]

        return result

    def save(self, filepath: str):
        """Save trained models to disk."""
        save_data = {
            'models': self.models,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        joblib.dump(save_data, filepath)
        logger.info(f"Models saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FakeNewsClassifier':
        """Load trained models from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.models = data['models']
        instance.metrics = data['metrics']
        instance.is_trained = data['is_trained']
        instance.feature_names = data.get('feature_names')
        logger.info(f"Models loaded from {filepath}")
        return instance

    def get_model_info(self) -> List[Dict]:
        """Get info about all available models."""
        info = []
        for name, config in self.MODEL_CONFIGS.items():
            info.append({
                'key': name,
                'display_name': config['display_name'],
                'description': config['description'],
                'is_trained': self.is_trained.get(name, False),
                'metrics': self.metrics.get(name, ModelMetrics()).to_dict()
                if self.is_trained.get(name, False) else {}
            })
        return info
