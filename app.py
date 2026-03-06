"""
Fake News Detection System - Streamlit Web App
Interactive UI for classifying news articles as FAKE or REAL
"""

import sys
import os
import logging
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessor import TextPreprocessor
from models import FakeNewsClassifier
from explainer import FakeNewsExplainer
from dataset import FakeNewsDataset

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .fake-badge {
        background: linear-gradient(135deg, #ff4757, #ff3838);
        color: white; padding: 8px 20px; border-radius: 25px;
        font-size: 1.4em; font-weight: bold; display: inline-block;
        box-shadow: 0 4px 15px rgba(255,71,87,0.4);
    }
    .real-badge {
        background: linear-gradient(135deg, #2ed573, #1e9e50);
        color: white; padding: 8px 20px; border-radius: 25px;
        font-size: 1.4em; font-weight: bold; display: inline-block;
        box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }
    .metric-card {
        background: #1e1e2e; border: 1px solid #333; border-radius: 12px;
        padding: 20px; text-align: center; margin: 5px;
    }
    div[data-testid="stMetric"] { background: #1e1e2e; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ────────────────────────────────────────────────
def init_session_state():
    defaults = {
        'classifier': None,
        'preprocessor': None,
        'explainer': None,
        'is_trained': False,
        'training_results': {},
        'prediction_history': [],
        'selected_model': 'logistic_regression'
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── Load / Train Models ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_models(use_demo: bool = True, demo_samples: int = 1000):
    """Load pre-trained models or train new ones on demo data."""
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        use_lemmatization=True,
        min_word_length=2
    )
    classifier = FakeNewsClassifier()

    model_path = Path('models/fake_news_classifier.joblib')

    if model_path.exists():
        try:
            classifier = FakeNewsClassifier.load(str(model_path))
            return preprocessor, classifier, {}, 'loaded'
        except Exception:
            pass

    # Train on demo dataset
    dataset = FakeNewsDataset(data_dir='data')
    df = dataset.create_demo_dataset(n_samples=demo_samples)
    df = dataset.preprocess_dataframe(preprocessor)

    X_train, X_test, y_train, y_test = dataset.split(df)

    train_metrics = classifier.train(X_train.tolist(), y_train, cv_folds=3)
    test_metrics = classifier.evaluate(X_test.tolist(), y_test)

    # Save
    model_path.parent.mkdir(exist_ok=True)
    classifier.save(str(model_path))

    results = {name: m.to_dict() for name, m in test_metrics.items()}
    return preprocessor, classifier, results, 'trained'


# ─── Confidence Gauge Chart ───────────────────────────────────────────────────
def create_confidence_gauge(confidence: float, label: str) -> go.Figure:
    color = "#ff4757" if label == "FAKE" else "#2ed573"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 32, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,71,87,0.1)'},
                {'range': [50, 75], 'color': 'rgba(255,165,0,0.1)'},
                {'range': [75, 100], 'color': 'rgba(46,213,115,0.1)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        },
        title={'text': 'Confidence', 'font': {'size': 18}}
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    return fig


# ─── Probability Bar Chart ────────────────────────────────────────────────────
def create_probability_bars(probabilities: dict) -> go.Figure:
    categories = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = ['#ff4757' if c == 'FAKE' else '#2ed573' for c in categories]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        width=0.4
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'range': [0, 110], 'showgrid': False, 'zeroline': False},
        xaxis={'showgrid': False},
        font_color='white', showlegend=False
    )
    return fig

# ─── Feature Importance Chart ─────────────────────────────────────────────────
def create_feature_chart(features: list, title: str, color: str) -> go.Figure:
    if not features:
        return None
    words = [f[0] for f in features[:10]]
    scores = [f[1] for f in features[:10]]
    fig = go.Figure(go.Bar(
        y=words, x=scores,
        orientation='h',
        marker_color=color,
        text=[f'{s:.3f}' for s in scores],
        textposition='outside'
    ))
    fig.update_layout(
        title=title, height=300,
        margin=dict(l=10, r=60, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'autorange': 'reversed', 'tickfont': {'size': 11}},
        xaxis={'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.1)'},
        font_color='white'
    )
    return fig


# ─── Model Comparison Chart ───────────────────────────────────────────────────
def create_model_comparison(results: dict) -> go.Figure:
    if not results:
        return None
    models = []
    for name, m in results.items():
        config = FakeNewsClassifier.MODEL_CONFIGS.get(name, {})
        models.append({
            'name': config.get('display_name', name),
            'accuracy': m.get('accuracy', 0) * 100,
            'f1': m.get('f1_score', 0) * 100,
            'roc_auc': m.get('roc_auc', 0) * 100,
        })
    df = pd.DataFrame(models)
    fig = go.Figure()
    metrics = ['accuracy', 'f1', 'roc_auc']
    colors = ['#5352ed', '#2ed573', '#ffa502']
    labels = ['Accuracy', 'F1 Score', 'ROC AUC']
    for metric, color, label in zip(metrics, colors, labels):
        fig.add_trace(go.Bar(
            name=label, x=df['name'], y=df[metric],
            marker_color=color, text=[f'{v:.1f}%' for v in df[metric]],
            textposition='outside'
        ))
    fig.update_layout(
        barmode='group', height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'range': [0, 115], 'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.1)'},
        legend={'bgcolor': 'rgba(0,0,0,0)'},
        font_color='white',
        title='Model Performance Comparison (%)'
    )
    return fig


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🔍 Fake News Detector")
        st.markdown("---")
        st.subheader("⚙️ Settings")
        model_options = {
            'logistic_regression': '📊 Logistic Regression',
            'naive_bayes': '🎯 Naive Bayes',
            'svm': '⚡ SVM'
        }
        selected = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        st.session_state.selected_model = selected
        st.markdown("---")
        st.subheader("🚀 Model Training")
        demo_samples = st.slider('Demo Samples', 500, 5000, 1000, step=500)
        if st.button("🔄 Train / Reload Models", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        st.markdown("---")
        st.subheader("📊 Dataset Info")
        st.info(
            "**Supported Datasets:**\n"
            "- ISOT (Fake.csv + True.csv)\n"
            "- WELFake (WELFake_Dataset.csv)\n"
            "- Any CSV with text+label cols\n\n"
            "Place files in **data/** folder.\n"
            "Demo mode uses synthetic data."
        )

    # Load Models
    with st.spinner("Loading / Training models..."):
        preprocessor, classifier, results, status = load_or_train_models(
            use_demo=True, demo_samples=demo_samples
        )
        st.session_state.classifier = classifier
        st.session_state.preprocessor = preprocessor
        st.session_state.explainer = FakeNewsExplainer(classifier, preprocessor)
        st.session_state.training_results = results
        st.session_state.is_trained = True

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("🔍 Fake News Detection System")
        st.markdown("*Classify news articles as **REAL** or **FAKE** using ML + Explainable AI*")
    with col_h2:
        if status == 'loaded':
            st.success("✅ Models Loaded")
        else:
            st.success("✅ Models Trained")
    st.markdown("---")

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Classify News", "📈 Model Performance", "🧠 Explainability", "📚 About"
    ])

    # TAB 1 - Classify
    with tab1:
        st.subheader("Paste or type a news article to classify")
        sample_fake = (
            "SHOCKING BOMBSHELL: Secret Government Program EXPOSED! "
            "Anonymous sources reveal that the deep state has been secretly "
            "hiding evidence of a massive conspiracy. Mainstream media REFUSES to cover this. "
            "Share before deleted! Wake up sheeple, they dont want you to know the truth."
        )
        sample_real = (
            "Federal Reserve raises interest rates by 25 basis points as inflation cools. "
            "The Federal Reserve announced it would raise its benchmark interest rate. "
            "Fed Chair stated that recent data shows inflation declining but remains above 2%. "
            "According to officials, the decision was unanimous among FOMC members."
        )
        col_s1, col_s2, col_s3 = st.columns(3)
        if col_s1.button('📋 Sample FAKE News', use_container_width=True):
            st.session_state['input_text'] = sample_fake
        if col_s2.button('📋 Sample REAL News', use_container_width=True):
            st.session_state['input_text'] = sample_real
        if col_s3.button('🗑️ Clear', use_container_width=True):
            st.session_state['input_text'] = ''

        text_input = st.text_area(
            "News Article Text:",
            value=st.session_state.get('input_text', ''),
            height=180,
            placeholder="Paste a news article here..."
        )

        analyze_btn = st.button('🔍 Analyze Article', type='primary', use_container_width=True)
        st.info(f"Using: **{model_options[st.session_state.selected_model]}**")

        if analyze_btn and text_input.strip():
            processed_text = preprocessor.preprocess(text_input)
            if len(processed_text.split()) < 3:
                st.error("⚠️ Text too short — please enter a longer article.")
            else:
                all_predictions = classifier.predict_all_models(processed_text)
                main_pred = classifier.predict(processed_text, st.session_state.selected_model)
                label = main_pred["label"]
                confidence = main_pred["confidence"]

                st.markdown("---")
                col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
                with col_r1:
                    st.markdown("### 🎯 Classification Result")
                    badge_class = "fake-badge" if label == "FAKE" else "real-badge"
                    icon = "⚠️" if label == "FAKE" else "✅"
                    st.markdown(f'<div class="{badge_class}">{icon} {label}</div>',
                                unsafe_allow_html=True)
                    st.markdown(f"**Model:** {main_pred['model']}")
                with col_r2:
                    st.plotly_chart(create_confidence_gauge(confidence, label),
                                    use_container_width=True)
                with col_r3:
                    if "probabilities" in main_pred:
                        st.plotly_chart(create_probability_bars(main_pred["probabilities"]),
                                        use_container_width=True)

                st.markdown("#### 🤖 All Models")
                model_cols = st.columns(len(all_predictions))
                for i, (mname, mpred) in enumerate(all_predictions.items()):
                    with model_cols[i]:
                        mcolor = "🔴" if mpred["label"] == "FAKE" else "🟢"
                        config = FakeNewsClassifier.MODEL_CONFIGS.get(mname, {})
                        st.metric(
                            label=config.get('display_name', mname),
                            value=f"{mcolor} {mpred['label']}",
                            delta=f"{mpred['confidence']:.0%} confident"
                        )

                explainer = st.session_state.explainer
                patterns = explainer.analyze_text_patterns(text_input)
                st.markdown("#### 📊 Text Pattern Analysis")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    stats = patterns.get("text_stats", {})
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Words", stats.get("word_count", 0))
                    c2.metric("Sentences", stats.get("sentence_count", 0))
                    c3.metric("CAPS Ratio", f"{stats.get('caps_ratio', 0):.1%}")
                    if patterns.get("fake_signals"):
                        st.warning("⚠️ **Fake News Signals Detected:**")
                        for cat, items in patterns["fake_signals"].items():
                            st.markdown(f"• **{cat.replace('_', ' ').title()}:** {str(items[:3])}")
                with col_p2:
                    if patterns.get("real_signals"):
                        st.success("✅ **Credibility Indicators Found:**")
                        for cat, items in patterns["real_signals"].items():
                            st.markdown(f"• **{cat.replace('_', ' ').title()}:** {str(items[:3])}")
                    else:
                        st.info("ℹ️ No strong credibility indicators found")

                contributions = explainer.get_word_contributions(
                    processed_text, st.session_state.selected_model
                )
                if contributions:
                    st.markdown("#### 🔤 Key Word Contributions")
                    col_wc1, col_wc2 = st.columns(2)
                    fake_words = [(w["word"], w["strength"]) for w in contributions
                                  if w["direction"] == "FAKE"][:10]
                    real_words = [(w["word"], w["strength"]) for w in contributions
                                  if w["direction"] == "REAL"][:10]
                    with col_wc1:
                        if fake_words:
                            fig = create_feature_chart(fake_words, "⚠️ Words → FAKE", "#ff4757")
                            if fig: st.plotly_chart(fig, use_container_width=True)
                    with col_wc2:
                        if real_words:
                            fig = create_feature_chart(real_words, "✅ Words → REAL", "#2ed573")
                            if fig: st.plotly_chart(fig, use_container_width=True)

                st.session_state.prediction_history.append({
                    'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                    'label': label,
                    'confidence': confidence,
                    'model': main_pred['model']
                })
        elif analyze_btn:
            st.warning("⚠️ Please enter some text first.")

        if st.session_state.prediction_history:
            st.markdown("---")
            st.subheader("📜 Prediction History")
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # TAB 2 - Performance
    with tab2:
        st.subheader("📈 Model Performance Metrics")
        if st.session_state.training_results:
            results = st.session_state.training_results
            fig_cmp = create_model_comparison(results)
            if fig_cmp: st.plotly_chart(fig_cmp, use_container_width=True)
            rows = []
            for name, m in results.items():
                config = FakeNewsClassifier.MODEL_CONFIGS.get(name, {})
                rows.append({
                    'Model': config.get('display_name', name),
                    'Accuracy': f"{m.get('accuracy', 0):.4f}",
                    'F1 Score': f"{m.get('f1_score', 0):.4f}",
                    'ROC AUC': f"{m.get('roc_auc', 0):.4f}"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Train models first to see performance metrics.")

    # TAB 3 - Explainability
    with tab3:
        st.subheader("🧠 Explainable AI — Feature Importance")
        xai_model = st.selectbox(
            "Select model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        if st.button("📊 Show Feature Importance", use_container_width=True):
            try:
                top_features = classifier.get_top_features(xai_model, n_features=20)
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    fake_feats = top_features.get("fake_features", [])
                    fig = create_feature_chart(fake_feats, "⚠️ Top FAKE-indicating Words", "#ff4757")
                    if fig: st.plotly_chart(fig, use_container_width=True)
                with col_f2:
                    real_feats = top_features.get("real_features", [])
                    fig = create_feature_chart(real_feats, "✅ Top REAL-indicating Words", "#2ed573")
                    if fig: st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not extract features: {e}")

    # TAB 4 - About
    with tab4:
        st.subheader("🎯 About This System")
        st.markdown("""
        This system classifies news as **REAL** or **FAKE** using:
        - **Logistic Regression** with L2 regularization
        - **Complement Naive Bayes** optimized for text classification
        - **Linear SVM** with probability calibration
        
        **Pipeline:** Text cleaning → Tokenization → Stopword removal → Lemmatization
        → TF-IDF → ML Model → Confidence Score + Explainability
        
        **Datasets:** ISOT (~44K articles), WELFake (~72K articles), or demo synthetic data.
        
        **Tech:** scikit-learn, NLTK, LIME, Streamlit, Plotly
        """)


if __name__ == "__main__":
    main()
