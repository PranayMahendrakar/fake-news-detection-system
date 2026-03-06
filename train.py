"""
Training Script for Fake News Detection System
Trains all three ML models and saves them for inference
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import FakeNewsDataset
from preprocessor import TextPreprocessor
from models import FakeNewsClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)


def ensure_dirs():
    """Create necessary directories."""
    for d in ['data', 'models', 'logs', 'results']:
        Path(d).mkdir(parents=True, exist_ok=True)


def train_pipeline(
    data_dir: str = 'data',
    model_path: str = 'models/fake_news_classifier.joblib',
    test_size: float = 0.2,
    cv_folds: int = 5,
    use_demo: bool = False,
    demo_samples: int = 2000
):
    """
    Full training pipeline:
    1. Load dataset (Kaggle or demo)
    2. Preprocess text
    3. Train all 3 models with cross-validation
    4. Evaluate on test set
    5. Save models and results
    """
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("Fake News Detection - Training Pipeline")
    logger.info("=" * 60)

    # ─── Step 1: Load Data ────────────────────────────────────────
    logger.info("Step 1: Loading dataset...")
    dataset = FakeNewsDataset(data_dir=data_dir)

    if use_demo:
        logger.info(f"Using demo dataset ({demo_samples} samples)")
        df = dataset.create_demo_dataset(n_samples=demo_samples)
    else:
        try:
            df = dataset.auto_detect_and_load()
        except FileNotFoundError as e:
            logger.warning(f"Kaggle dataset not found: {e}")
            logger.warning("Falling back to demo dataset...")
            df = dataset.create_demo_dataset(n_samples=demo_samples)

    stats = dataset.get_stats(df)
    logger.info(f"Dataset stats: {json.dumps(stats, indent=2)}")

    # ─── Step 2: Preprocess ───────────────────────────────────────
    logger.info("Step 2: Preprocessing text...")
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        use_lemmatization=True,
        min_word_length=2
    )

    cache_path = 'data/preprocessed_cache.parquet'
    df = dataset.preprocess_dataframe(
        preprocessor,
        text_column='full_text',
        cache_path=cache_path if not use_demo else None
    )

    # ─── Step 3: Split ────────────────────────────────────────────
    logger.info("Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = dataset.split(
        df,
        test_size=test_size,
        text_column='processed_text'
    )

    # ─── Step 4: Train Models ─────────────────────────────────────
    logger.info("Step 4: Training models...")
    classifier = FakeNewsClassifier()

    train_metrics = classifier.train(
        X_train.tolist(),
        y_train,
        cv_folds=cv_folds
    )

    logger.info("Training metrics:")
    for model_name, metrics in train_metrics.items():
        logger.info(
            f"  {model_name}: CV F1={metrics.cv_mean:.4f}±{metrics.cv_std:.4f}"
        )

    # ─── Step 5: Evaluate on Test Set ────────────────────────────
    logger.info("Step 5: Evaluating on test set...")
    test_metrics = classifier.evaluate(X_test.tolist(), y_test)

    results = {}
    logger.info("Test set metrics:")
    for model_name, metrics in test_metrics.items():
        m_dict = metrics.to_dict()
        results[model_name] = m_dict
        logger.info(
            f"  {model_name}: Acc={m_dict['accuracy']:.4f}, "
            f"F1={m_dict['f1_score']:.4f}, AUC={m_dict['roc_auc']:.4f}"
        )
        logger.info(f"  Classification Report:\n{metrics.classification_report}")

    # ─── Step 6: Save Models ──────────────────────────────────────
    logger.info(f"Step 6: Saving models to {model_path}...")
    classifier.save(model_path)

    # Save results
    results_path = 'results/training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ─── Find Best Model ──────────────────────────────────────────
    best_model = max(results, key=lambda k: results[k]['f1_score'])
    logger.info(
        f"\n{'=' * 60}\n"
        f"Best model: {best_model}\n"
        f"  F1 Score: {results[best_model]['f1_score']:.4f}\n"
        f"  Accuracy: {results[best_model]['accuracy']:.4f}\n"
        f"  ROC AUC:  {results[best_model]['roc_auc']:.4f}\n"
        f"{'=' * 60}"
    )

    return classifier, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fake News Detection Models')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--model-path', default='models/fake_news_classifier.joblib',
                        help='Path to save trained models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--demo', action='store_true',
                        help='Use demo dataset (no Kaggle download needed)')
    parser.add_argument('--demo-samples', type=int, default=2000,
                        help='Number of demo samples to generate')
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        model_path=args.model_path,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        use_demo=args.demo,
        demo_samples=args.demo_samples
    )
