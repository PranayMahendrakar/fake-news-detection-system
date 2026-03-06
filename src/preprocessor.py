"""
Text Preprocessor for Fake News Detection System
Handles cleaning, tokenization, stopword removal, and stemming/lemmatization
"""

import re
import string
import logging
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logging.warning(f"Could not download {resource}: {e}")

download_nltk_resources()

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessor for news article classification.
    
    Features:
    - HTML/URL removal
    - Lowercasing
    - Punctuation removal
    - Stopword removal
    - Lemmatization or stemming
    - Noise word removal
    """

    # Common noise patterns in news articles
    NOISE_PATTERNS = [
        r'http\S+|www\.\S+',          # URLs
        r'<[^>]+>',                        # HTML tags
        r'\[.*?\]',                      # Square brackets
        r'\(.*?\)',                      # Parenthetical citations
        r'\b\d+\b',                     # Standalone numbers
        r'[^\w\s]',                      # Special characters
        r'\s+',                           # Multiple whitespaces
    ]

    def __init__(
        self,
        remove_stopwords: bool = True,
        use_lemmatization: bool = True,
        min_word_length: int = 2,
        max_word_length: int = 50,
        language: str = 'english'
    ):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.language = language

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        # Load stopwords
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(language))

        # Additional custom stopwords for news
        self.custom_stopwords = {
            'said', 'says', 'told', 'according', 'reported', 'would',
            'could', 'also', 'one', 'two', 'three', 'new', 'year',
            'time', 'day', 'week', 'month', 'ago', 'like', 'get', 'make'
        }
        self.stop_words.update(self.custom_stopwords)

    def clean_text(self, text: str) -> str:
        """Remove noise patterns from text."""
        if not isinstance(text, str):
            text = str(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens based on length and stopwords."""
        filtered = []
        for token in tokens:
            # Check length constraints
            if len(token) < self.min_word_length or len(token) > self.max_word_length:
                continue

            # Remove stopwords if enabled
            if self.remove_stopwords and token in self.stop_words:
                continue

            # Keep only alphabetic tokens
            if not token.isalpha():
                continue

            filtered.append(token)
        return filtered

    def normalize(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization or stemming."""
        if self.use_lemmatization:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            return [self.stemmer.stem(token) for token in tokens]

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.
        
        Pipeline:
        1. Clean text (remove noise)
        2. Tokenize
        3. Filter tokens
        4. Normalize (lemmatize/stem)
        5. Rejoin into string
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Clean
        cleaned = self.clean_text(text)

        # Step 2: Tokenize
        tokens = self.tokenize(cleaned)

        # Step 3: Filter
        filtered = self.filter_tokens(tokens)

        # Step 4: Normalize
        normalized = self.normalize(filtered)

        # Step 5: Rejoin
        return ' '.join(normalized)

    def preprocess_batch(self, texts: List[str], verbose: bool = False) -> List[str]:
        """Process a batch of texts."""
        results = []
        total = len(texts)
        for i, text in enumerate(texts):
            if verbose and i % 1000 == 0:
                logger.info(f"Preprocessing: {i}/{total}")
            results.append(self.preprocess(text))
        return results

    def get_stats(self, text: str) -> dict:
        """Get preprocessing statistics for a text."""
        original_tokens = text.split()
        processed = self.preprocess(text)
        processed_tokens = processed.split()

        return {
            'original_word_count': len(original_tokens),
            'processed_word_count': len(processed_tokens),
            'reduction_ratio': 1 - (len(processed_tokens) / max(len(original_tokens), 1)),
            'original_char_count': len(text),
            'processed_char_count': len(processed),
            'unique_words': len(set(processed_tokens))
        }
