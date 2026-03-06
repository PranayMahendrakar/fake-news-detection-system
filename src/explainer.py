"""
Explainable AI Module for Fake News Detection
Uses LIME for local explanations and custom feature importance analysis
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class FakeNewsExplainer:
    """
    Provides explainability for fake news predictions using multiple techniques:
    
    1. LIME (Local Interpretable Model-agnostic Explanations)
       - Local linear approximation around each prediction
       - Highlights which words pushed toward FAKE or REAL
       
    2. Feature Importance
       - Global model-level word importance from coefficients
       - Shows most discriminative features overall
       
    3. Keyword Highlighting
       - Marks suspicious/credible patterns in the original text
       - Rule-based detection of common fake news indicators
    """

    # Common patterns associated with fake/misinformation news
    FAKE_INDICATORS = {
        'sensational_words': [
            'shocking', 'unbelievable', 'bombshell', 'explosive', 'breaking',
            'exposed', 'scandal', 'outrage', 'secret', 'banned', 'censored',
            'they dont want you to know', 'wake up', 'share before deleted',
            'mainstream media', 'deep state', 'hoax', 'false flag'
        ],
        'emotional_manipulation': [
            'urgent', 'must read', 'must share', 'spread the word',
            'please share', 'going viral', 'everyone needs to see',
            'you wont believe', 'jaw dropping'
        ],
        'source_problems': [
            'anonymous source', 'sources say', 'rumored', 'allegedly',
            'unconfirmed', 'could be', 'some say', 'many people are saying'
        ],
        'absolute_claims': [
            'always', 'never', 'every single', 'all of them', 'no one',
            'completely', 'totally', 'absolutely', 'definitely', '100%'
        ]
    }

    # Patterns associated with credible/real news
    REAL_INDICATORS = {
        'attribution': [
            'according to', 'said in a statement', 'confirmed by', 'reported by',
            'study shows', 'research indicates', 'data shows', 'statistics show',
            'officials confirmed', 'spokesperson said'
        ],
        'qualifications': [
            'however', 'although', 'on the other hand', 'it should be noted',
            'experts disagree', 'this is disputed', 'in context', 'evidence suggests'
        ],
        'specific_details': [
            'percent', 'according to', 'published in', 'peer-reviewed',
            'journal of', 'university of', 'department of', 'institute'
        ]
    }

    def __init__(self, classifier=None, preprocessor=None):
        self.classifier = classifier
        self.preprocessor = preprocessor
        self._lime_explainer = None

    def _get_lime_explainer(self, class_names: List[str] = None):
        """Lazy-load LIME explainer."""
        if self._lime_explainer is None:
            try:
                from lime.lime_text import LimeTextExplainer
                self._lime_explainer = LimeTextExplainer(
                    class_names=class_names or ['FAKE', 'REAL'],
                    random_state=42
                )
            except ImportError:
                logger.warning("LIME not available. Install with: pip install lime")
                return None
        return self._lime_explainer

    def explain_with_lime(
        self,
        text: str,
        model_name: str = 'logistic_regression',
        num_features: int = 15,
        num_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single prediction.
        
        Returns:
            {
                'features': List of (word, weight) sorted by importance,
                'fake_features': Words pushing toward FAKE,
                'real_features': Words pushing toward REAL,
                'prediction': model prediction result,
                'lime_available': bool
            }
        """
        if self.classifier is None:
            return {'lime_available': False, 'error': 'No classifier loaded'}

        explainer = self._get_lime_explainer()
        if explainer is None:
            return {'lime_available': False, 'error': 'LIME not installed'}

        try:
            pipeline = self.classifier.models[model_name]

            def predict_proba(texts):
                if hasattr(pipeline, 'predict_proba'):
                    return pipeline.predict_proba(texts)
                else:
                    preds = pipeline.predict(texts)
                    proba = np.zeros((len(preds), 2))
                    proba[preds == 0, 0] = 1.0
                    proba[preds == 1, 1] = 1.0
                    return proba

            # Generate explanation
            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )

            features = exp.as_list()
            fake_features = [(w, abs(wt)) for w, wt in features if wt < 0]
            real_features = [(w, abs(wt)) for w, wt in features if wt >= 0]

            return {
                'lime_available': True,
                'features': sorted(features, key=lambda x: abs(x[1]), reverse=True),
                'fake_features': sorted(fake_features, key=lambda x: x[1], reverse=True),
                'real_features': sorted(real_features, key=lambda x: x[1], reverse=True),
                'prediction_local': exp.predict_proba.tolist(),
                'intercept': exp.intercept[1] if hasattr(exp, 'intercept') else None
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'lime_available': False, 'error': str(e)}

    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """
        Rule-based analysis of text patterns to detect fake news indicators.
        
        Returns detailed breakdown of suspicious vs credible patterns found.
        """
        text_lower = text.lower()
        results = {
            'fake_signals': {},
            'real_signals': {},
            'fake_score': 0,
            'real_score': 0,
            'highlighted_words': []
        }

        total_fake = 0
        total_real = 0

        # Check fake indicators
        for category, patterns in self.FAKE_INDICATORS.items():
            found = []
            for pattern in patterns:
                if pattern in text_lower:
                    found.append(pattern)
                    total_fake += 1
            if found:
                results['fake_signals'][category] = found

        # Check real indicators
        for category, patterns in self.REAL_INDICATORS.items():
            found = []
            for pattern in patterns:
                if pattern in text_lower:
                    found.append(pattern)
                    total_real += 1
            if found:
                results['real_signals'][category] = found

        # Calculate signal scores
        total = total_fake + total_real
        if total > 0:
            results['fake_score'] = round(total_fake / total, 3)
            results['real_score'] = round(total_real / total, 3)
        else:
            results['fake_score'] = 0.5
            results['real_score'] = 0.5

        # Text statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        results['text_stats'] = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': round(np.mean([len(w) for w in words]) if words else 0, 2),
            'caps_ratio': round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 3),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_numeric_claims': bool(re.search(r'\d+%|\$\d+|\d+ million|\d+ billion', text))
        }

        return results

    def get_word_contributions(
        self,
        text: str,
        model_name: str = 'logistic_regression'
    ) -> List[Dict[str, Any]]:
        """
        Get per-word contribution to the prediction by checking model features.
        
        Returns list of {word, contribution, direction} for words in text.
        """
        if self.classifier is None or not self.classifier.is_trained.get(model_name, False):
            return []

        try:
            pipeline = self.classifier.models[model_name]
            vectorizer = pipeline.named_steps['vectorizer']
            classifier = pipeline.named_steps['classifier']

            # Get feature names and coefficients
            feature_names = list(vectorizer.get_feature_names_out())

            coef = None
            if hasattr(classifier, 'coef_'):
                coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
            elif hasattr(classifier, 'estimator') and hasattr(classifier.estimator, 'coef_'):
                coef = classifier.estimator.coef_[0]
            elif hasattr(classifier, 'feature_log_prob_'):
                # Naive Bayes: real - fake log probs
                coef = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

            if coef is None:
                return []

            feature_dict = {name: float(c) for name, c in zip(feature_names, coef)}

            # Preprocess and get words
            if self.preprocessor:
                processed = self.preprocessor.preprocess(text)
            else:
                processed = text.lower()

            words = processed.split()
            word_contributions = []

            for word in set(words):
                if word in feature_dict:
                    contrib = feature_dict[word]
                    word_contributions.append({
                        'word': word,
                        'contribution': round(float(contrib), 4),
                        'direction': 'REAL' if contrib > 0 else 'FAKE',
                        'strength': abs(round(float(contrib), 4))
                    })

            # Sort by absolute contribution
            word_contributions.sort(key=lambda x: x['strength'], reverse=True)
            return word_contributions[:30]  # Top 30 contributing words

        except Exception as e:
            logger.error(f"Word contribution analysis failed: {e}")
            return []

    def generate_explanation_summary(
        self,
        text: str,
        prediction: Dict[str, Any],
        model_name: str = 'logistic_regression'
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation combining all methods.
        
        Returns:
            Full explanation with lime, patterns, word contributions,
            and a human-readable summary
        """
        summary = {
            'prediction': prediction,
            'lime_explanation': {},
            'pattern_analysis': {},
            'word_contributions': [],
            'human_summary': '',
            'confidence_factors': []
        }

        # LIME explanation
        if self.classifier and self.classifier.is_trained.get(model_name, False):
            summary['lime_explanation'] = self.explain_with_lime(
                text, model_name, num_features=10
            )
            summary['word_contributions'] = self.get_word_contributions(
                text, model_name
            )

        # Pattern analysis
        summary['pattern_analysis'] = self.analyze_text_patterns(text)

        # Generate human-readable summary
        label = prediction.get('label', 'UNKNOWN')
        confidence = prediction.get('confidence', 0.0)

        factors = []

        if confidence > 0.9:
            factors.append(f"Very high confidence ({confidence:.0%}) in {label} classification")
        elif confidence > 0.75:
            factors.append(f"High confidence ({confidence:.0%}) that this is {label}")
        else:
            factors.append(f"Moderate confidence ({confidence:.0%}), result may be uncertain")

        pattern = summary['pattern_analysis']
        if pattern.get('fake_signals'):
            sig_count = sum(len(v) for v in pattern['fake_signals'].values())
            factors.append(f"Found {sig_count} fake news language pattern(s)")
        if pattern.get('real_signals'):
            sig_count = sum(len(v) for v in pattern['real_signals'].values())
            factors.append(f"Found {sig_count} credibility indicator(s)")

        stats = pattern.get('text_stats', {})
        if stats.get('caps_ratio', 0) > 0.15:
            factors.append("High use of capital letters (common in sensational content)")
        if stats.get('exclamation_count', 0) > 3:
            factors.append(f"Excessive exclamation marks ({stats['exclamation_count']})")

        summary['confidence_factors'] = factors
        summary['human_summary'] = f"This article is classified as {label}. " + ". ".join(factors[:3]) + "."

        return summary
