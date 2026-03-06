"""
Dataset Loader for Fake News Detection
Supports Kaggle Fake News Dataset (ISOT, WELFake, or similar)
Handles loading, merging, splitting, and validation
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FakeNewsDataset:
    """
    Dataset manager for Fake News Detection.
    
    Supports multiple popular Kaggle fake news datasets:
    
    1. ISOT Dataset (Fake.csv + True.csv)
       - Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
       - ~44,000 articles
       
    2. WELFake Dataset (WELFake_Dataset.csv)
       - Source: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
       - ~72,000 articles
       
    3. Generic (fake_news.csv with label column)
    
    Dataset schema after loading:
        - text: full article text (title + body)
        - label: 0 = FAKE, 1 = REAL
        - title: article title
        - subject: news subject/category (if available)
    """

    LABEL_MAP = {'fake': 0, 'real': 1, '0': 0, '1': 1, 0: 0, 1: 1}

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.raw_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self._stats: Dict[str, Any] = {}

    def load_isot(
        self,
        fake_path: Optional[str] = None,
        true_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load ISOT Fake News Dataset (two separate CSV files).
        
        Args:
            fake_path: Path to Fake.csv
            true_path: Path to True.csv
            
        Expected columns: title, text, subject, date
        """
        fake_path = fake_path or self.data_dir / 'Fake.csv'
        true_path = true_path or self.data_dir / 'True.csv'

        logger.info(f"Loading ISOT dataset from {fake_path} and {true_path}")

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df['label'] = 0  # FAKE
        true_df['label'] = 1  # REAL

        df = pd.concat([fake_df, true_df], ignore_index=True)

        # Create combined text feature
        if 'title' in df.columns and 'text' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'text' in df.columns:
            df['full_text'] = df['text'].fillna('')
        elif 'title' in df.columns:
            df['full_text'] = df['title'].fillna('')

        self.raw_df = df
        logger.info(f"Loaded {len(df)} samples: {(df.label == 0).sum()} fake, {(df.label == 1).sum()} real")
        return df

    def load_welfake(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load WELFake Dataset (single CSV with label column).
        
        Expected columns: Unnamed: 0, title, text, label (0=fake, 1=real)
        """
        filepath = filepath or self.data_dir / 'WELFake_Dataset.csv'
        logger.info(f"Loading WELFake dataset from {filepath}")

        df = pd.read_csv(filepath)

        # Standardize label column
        if 'label' not in df.columns:
            raise ValueError("Dataset must have a 'label' column")

        df['label'] = df['label'].map(self.LABEL_MAP).fillna(df['label'])
        df['label'] = df['label'].astype(int)

        if 'title' in df.columns and 'text' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'text' in df.columns:
            df['full_text'] = df['text'].fillna('')

        self.raw_df = df
        logger.info(f"Loaded {len(df)} samples")
        return df

    def load_generic(self, filepath: str, text_col: str = 'text', label_col: str = 'label') -> pd.DataFrame:
        """
        Load any CSV dataset with configurable column names.
        """
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found. Available: {list(df.columns)}")

        df = df.rename(columns={text_col: 'full_text', label_col: 'label'})

        # Normalize labels
        label_lower = df['label'].astype(str).str.lower()
        label_map = {'fake': 0, 'false': 0, '0': 0, 'real': 1, 'true': 1, '1': 1}
        df['label'] = label_lower.map(label_map).fillna(pd.to_numeric(df['label'], errors='coerce'))
        df['label'] = df['label'].fillna(0).astype(int)

        self.raw_df = df
        return df

    def auto_detect_and_load(self) -> pd.DataFrame:
        """
        Auto-detect dataset format in data directory and load appropriately.
        """
        data_dir = self.data_dir

        # Check for ISOT format
        if (data_dir / 'Fake.csv').exists() and (data_dir / 'True.csv').exists():
            logger.info("Detected ISOT dataset format")
            return self.load_isot()

        # Check for WELFake
        if (data_dir / 'WELFake_Dataset.csv').exists():
            logger.info("Detected WELFake dataset format")
            return self.load_welfake()

        # Check for generic CSV
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            filepath = csv_files[0]
            df_preview = pd.read_csv(filepath, nrows=5)
            logger.info(f"Detected CSV: {filepath}, columns: {list(df_preview.columns)}")
            return self.load_generic(str(filepath))

        raise FileNotFoundError(
            f"No dataset found in {data_dir}.\n"
            "Please download a dataset from Kaggle:\n"
            "- ISOT: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "- WELFake: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification"
        )

    def create_demo_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a small demo dataset for testing/development without Kaggle download.
        Uses synthetic examples based on realistic patterns.
        """
        np.random.seed(42)

        fake_titles = [
            "SHOCKING: Secret Government Program EXPOSED!",
            "You Won't Believe What They're Hiding From You",
            "BREAKING: Bombshell Revelation Changes Everything",
            "Doctors HATE This! Amazing Cure Found",
            "URGENT: Share Before This Gets Deleted",
            "Mainstream Media REFUSES to Cover This Story",
            "Anonymous Source Reveals Deep State Conspiracy",
            "VIRAL: Unbelievable Truth About Vaccines",
            "Celebrity Shockingly Admits to Secret Society",
            "Scientists BANNED For Revealing Climate Lie",
        ]

        real_titles = [
            "Federal Reserve Raises Interest Rates by 0.25 Points",
            "Study Shows Mediterranean Diet Reduces Heart Disease Risk",
            "City Council Approves New Infrastructure Budget",
            "University Research Finds Link Between Exercise and Memory",
            "Congressional Committee Holds Hearing on Data Privacy",
            "WHO Reports Decline in Global Malaria Cases",
            "Tech Company Reports Quarterly Earnings Above Expectations",
            "Scientists Publish Peer-Reviewed Climate Study in Nature",
            "State Department Issues Travel Advisory for Region",
            "Hospital System Implements New Electronic Health Records",
        ]

        fake_texts = [
            "According to anonymous sources who cannot be named, the government has been secretly {claim}. "
            "Mainstream media refuses to cover this SHOCKING revelation. Share before it's deleted! "
            "They don't want you to know about this bombshell that changes everything we thought we knew.",
            "BREAKING NEWS that the corrupt establishment is hiding from you! Our insider sources reveal "
            "that {claim}. This explosive information has been CENSORED by big tech and mainstream media. "
            "Wake up sheeple! The truth is out there if you just open your eyes!",
        ]

        real_texts = [
            "According to officials at the department, {statement}. The announcement was confirmed by "
            "a spokesperson on Tuesday. Experts in the field noted that the data indicates a significant "
            "shift in policy. The report, published in the peer-reviewed journal, supports these findings.",
            "A new study published by researchers at the university found that {finding}. "
            "The research, which analyzed data from over 10,000 participants, was peer-reviewed before "
            "publication. Dr. Smith, lead author, said in a statement that the results were consistent "
            "with previous research in the field.",
        ]

        claims = [
            "hiding evidence of alien contact", "controlling the weather",
            "adding chemicals to water supplies", "planning a global takeover",
            "faking moon landing footage", "suppressing cancer cures"
        ]
        statements = [
            "the new policy will take effect next quarter",
            "funding has been allocated for the initiative",
            "the committee will review the proposal in March",
            "international cooperation on the matter is ongoing"
        ]
        findings = [
            "regular exercise improves cognitive function",
            "dietary changes reduce inflammation markers",
            "early intervention improves patient outcomes",
            "renewable energy costs have decreased significantly"
        ]

        rows = []
        half = n_samples // 2

        for i in range(half):
            title = fake_titles[i % len(fake_titles)]
            text_template = fake_texts[i % len(fake_texts)]
            text = text_template.format(claim=claims[i % len(claims)])
            rows.append({
                'full_text': title + ' ' + text,
                'title': title,
                'text': text,
                'label': 0,
                'subject': 'politics'
            })

        for i in range(half):
            title = real_titles[i % len(real_titles)]
            text_template = real_texts[i % len(real_texts)]
            text = text_template.format(
                statement=statements[i % len(statements)],
                finding=findings[i % len(findings)]
            )
            rows.append({
                'full_text': title + ' ' + text,
                'title': title,
                'text': text,
                'label': 1,
                'subject': 'politics'
            })

        df = pd.DataFrame(rows)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.raw_df = df
        logger.info(f"Created demo dataset with {len(df)} samples")
        return df

    def preprocess_dataframe(
        self,
        preprocessor,
        text_column: str = 'full_text',
        cache_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply text preprocessing to the dataframe.
        Optionally caches results to avoid re-preprocessing.
        """
        if self.raw_df is None:
            raise ValueError("No data loaded. Call load_*() first.")

        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading preprocessed data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info("Preprocessing text data...")
        df = self.raw_df.copy()
        df['processed_text'] = preprocessor.preprocess_batch(
            df[text_column].fillna('').tolist(),
            verbose=True
        )

        # Remove empty rows after preprocessing
        df = df[df['processed_text'].str.len() > 10].reset_index(drop=True)

        if cache_path:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached preprocessed data to {cache_path}")

        return df

    def split(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42,
        text_column: str = 'processed_text'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train/test sets.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        df = df if df is not None else self.raw_df
        if df is None:
            raise ValueError("No data available. Load data first.")

        # Fall back to full_text if processed_text not available
        if text_column not in df.columns:
            text_column = 'full_text' if 'full_text' in df.columns else 'text'

        # Remove NaN
        df = df.dropna(subset=[text_column, 'label'])

        X = df[text_column].values
        y = df['label'].values

        stratify_col = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        logger.info(
            f"Split: {len(X_train)} train, {len(X_test)} test | "
            f"Train FAKE: {(y_train == 0).sum()}, REAL: {(y_train == 1).sum()} | "
            f"Test FAKE: {(y_test == 0).sum()}, REAL: {(y_test == 1).sum()}"
        )

        return X_train, X_test, y_train, y_test

    def get_stats(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get dataset statistics."""
        df = df if df is not None else self.raw_df
        if df is None:
            return {}

        text_col = 'full_text' if 'full_text' in df.columns else 'text'
        word_counts = df[text_col].fillna('').str.split().str.len()

        stats = {
            'total_samples': len(df),
            'fake_count': int((df['label'] == 0).sum()),
            'real_count': int((df['label'] == 1).sum()),
            'class_balance': round((df['label'] == 0).sum() / len(df), 3),
            'avg_word_count': round(float(word_counts.mean()), 1),
            'min_word_count': int(word_counts.min()),
            'max_word_count': int(word_counts.max()),
            'median_word_count': round(float(word_counts.median()), 1),
            'columns': list(df.columns),
        }

        if 'subject' in df.columns:
            stats['subjects'] = df['subject'].value_counts().head(10).to_dict()

        self._stats = stats
        return stats

    def sample_articles(self, n: int = 5, label: Optional[int] = None) -> pd.DataFrame:
        """Get sample articles for display."""
        df = self.raw_df
        if df is None:
            return pd.DataFrame()

        if label is not None:
            df = df[df['label'] == label]

        return df.sample(min(n, len(df))).reset_index(drop=True)
