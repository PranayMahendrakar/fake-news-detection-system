# 🔍 Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Classify news articles as **REAL** or **FAKE** using machine learning with explainable AI, confidence scoring, and an interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **News Classification** | Binary classification: FAKE (0) or REAL (1) |
| **Text Preprocessing** | URL/HTML removal, tokenization, stopwords, lemmatization |
| **Confidence Score** | Probability-calibrated confidence for every prediction |
| **Explainable AI** | LIME explanations + feature coefficients + pattern analysis |
| **Multi-Model** | Compare Logistic Regression, Naive Bayes, and SVM simultaneously |
| **Pattern Detection** | Rule-based detection of fake news language patterns |
| **Interactive UI** | Streamlit dashboard with visualizations and history |

---

## 🤖 Models

### 1. Logistic Regression
- L2 regularization, balanced class weights
- Solver: `lbfgs`, max 1000 iterations
- TF-IDF with 100K features + bigrams
- ✅ Best interpretability — coefficients directly map to word importance

### 2. Naive Bayes (Complement NB)
- Complement Naive Bayes — optimized for imbalanced text datasets
- Smoothing: α = 0.1
- ✅ Fastest training and inference

### 3. Support Vector Machine (SVM)
- Linear SVC with `CalibratedClassifierCV` for probability scores
- TF-IDF with 100K features + trigrams
- Balanced class weights
- ✅ Highest accuracy on large datasets

---

## 📦 Dataset

### Kaggle Fake News Datasets (Recommended)

| Dataset | Articles | Format | Link |
|---------|----------|--------|------|
| **ISOT** | ~44,000 | `Fake.csv` + `True.csv` | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| **WELFake** | ~72,000 | `WELFake_Dataset.csv` | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |

### Demo Mode (No Download Required)
Synthetic dataset generator creates realistic fake/real news patterns for immediate testing.

**Expected accuracy:** ~95%+ on Kaggle data, ~70–80% on demo synthetic data.

---

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/PranayMahendrakar/fake-news-detection-system.git
cd fake-news-detection-system
pip install -r requirements.txt
```

### 2. Download NLTK Resources
```python
python -c "import nltk; nltk.download('all')"
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
The app trains models automatically on first launch using the demo dataset.

### 4. Use Real Kaggle Data (Optional)
```bash
# Download ISOT dataset from Kaggle and place in data/
mkdir -p data
# Place Fake.csv and True.csv in data/ folder

# Train with real data
python train.py --data-dir data/

# Or use WELFake
# Place WELFake_Dataset.csv in data/
python train.py --data-dir data/
```

---

## 📂 Project Structure

```
fake-news-detection-system/
├── app.py                    # Streamlit web application
├── train.py                  # Training script (CLI)
├── requirements.txt          # Dependencies
│
├── src/
│   ├── __init__.py           # Package init
│   ├── preprocessor.py       # Text preprocessing pipeline
│   ├── models.py             # ML models (LR, NB, SVM)
│   ├── explainer.py          # LIME + feature importance
│   └── dataset.py            # Dataset loader (ISOT, WELFake)
│
├── data/                     # Place Kaggle CSVs here
├── models/                   # Saved trained models
├── results/                  # Training metrics JSON
└── logs/                     # Training logs
```

---

## 🔍 Text Preprocessing Pipeline

```
Raw Text
  ↓  Remove URLs, HTML tags, special chars
  ↓  Lowercase
  ↓  Tokenize (NLTK punkt)
  ↓  Remove stopwords (NLTK + custom 15 words)
  ↓  Filter by length (2–50 chars)
  ↓  Lemmatize (WordNet) or Stem (Porter)
  ↓  Rejoin as string
Processed Text → TF-IDF → ML Model
```

---

## 🧠 Explainable AI

Three complementary explainability methods:

**1. LIME (Local Interpretable Model-agnostic Explanations)**
- Creates local linear approximations around each prediction
- Shows exactly which words pushed toward FAKE or REAL
- Works with all three models

**2. Feature Coefficient Analysis**
- Global view of most discriminative vocabulary
- Logistic Regression / SVM: positive coef → REAL, negative → FAKE
- Naive Bayes: log probability ratio between classes

**3. Rule-Based Pattern Detection**
```
Fake Signals:  sensational_words, emotional_manipulation, source_problems, absolute_claims
Real Signals:  attribution, qualifications, specific_details
```

---

## 🖥️ Streamlit UI

| Tab | Content |
|-----|---------|
| 🔍 **Classify News** | Text input, predictions, confidence gauge, word contributions |
| 📈 **Model Performance** | Accuracy/F1/AUC charts, metrics table |
| 🧠 **Explainability** | Feature importance charts, top FAKE/REAL words |
| 📚 **About** | System info, dataset guide, tech stack |

---

## ⚡ CLI Training

```bash
# Train with demo data
python train.py --demo --demo-samples 5000

# Train with Kaggle data
python train.py --data-dir data/ --test-size 0.2 --cv-folds 5

# Custom model save path
python train.py --model-path models/my_model.joblib
```

---

## 📊 Expected Performance

On ISOT Dataset (44K articles):

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~98.5% | ~98.5% | ~99.8% |
| Naive Bayes | ~96.0% | ~95.9% | ~99.2% |
| SVM | ~99.0% | ~99.0% | ~99.9% |

> Note: High accuracy on ISOT reflects dataset-specific patterns. Real-world accuracy varies.

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| ML Models | `scikit-learn` |
| Text Processing | `NLTK`, `regex` |
| Feature Extraction | `TfidfVectorizer` |
| Explainability | `LIME`, `SHAP` |
| Visualization | `Plotly`, `Seaborn`, `WordCloud` |
| Web UI | `Streamlit` |
| Model Persistence | `joblib` |
| Data Handling | `pandas`, `numpy` |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push and create a Pull Request

---

<p align="center">
Built with ❤️ by <a href="https://github.com/PranayMahendrakar">PranayMahendrakar</a>
</p>
