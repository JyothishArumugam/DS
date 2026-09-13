"""
Tweet Sentiment Classifier — Streamlit App
-------------------------------------------
Loads the BiLSTM model + tokenizer + label map saved by the companion notebook
(sentiment_classifier_workshop.ipynb, Section 9) and predicts sentiment for
user-entered text.

Run with:
    pip install streamlit tensorflow
    streamlit run app.py

Expects these files in the same folder (all produced by the notebook):
    sentiment_model.h5
    tokenizer.pickle
    label_encoder.pickle
    config.pickle
"""

import re
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tweet Sentiment Classifier", page_icon="🐦", layout="centered")

# ---------------------------------------------------------------------------
# Preprocessing — MUST mirror clean_text_dl() from the notebook exactly.
# If you change the cleaning logic in the notebook, update it here too, or
# your predictions will silently drift from what the model was trained on.
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
ELONGATED_RE = re.compile(r"(.)\1{2,}")

try:
    import contractions
    HAS_CONTRACTIONS = True
except ImportError:
    HAS_CONTRACTIONS = False


def clean_text_dl(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)
    if HAS_CONTRACTIONS:
        text = contractions.fix(text)
    text = ELONGATED_RE.sub(r"\1\1", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Cached loaders — everything here runs once per session, not per prediction.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    from tensorflow.keras.models import load_model

    model = load_model("sentiment_model.h5")

    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pickle", "rb") as f:
        label_map = pickle.load(f)

    with open("config.pickle", "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, label_map, config


def predict_sentiment(raw_text, model, tokenizer, label_map, config):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text_dl(raw_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=config["maxlen"], padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "cleaned": cleaned,
        "sentiment": label_map[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {label_map[i]: float(p) for i, p in enumerate(probs)},
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🐦 Tweet Sentiment Classifier")
st.caption("BiLSTM model trained in the companion notebook — enter any text below.")

try:
    model, tokenizer, label_map, config = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model artifacts not found. Run Section 9 of "
        "`sentiment_classifier_workshop.ipynb` first to generate "
        "`sentiment_model.h5`, `tokenizer.pickle`, `label_encoder.pickle`, "
        "and `config.pickle` in this folder.\n\n"
        f"Details: {e}"
    )
    st.stop()

SENTIMENT_STYLE = {
    "Positive": ("🟢", "#2ecc71"),
    "Neutral": ("⚪", "#95a5a6"),
    "Negative": ("🔴", "#e74c3c"),
}

user_text = st.text_area(
    "Enter a tweet or any short text:",
    placeholder="e.g. This new update is amazing, so much faster now!",
    height=100,
)

col1, col2 = st.columns([1, 4])
predict_clicked = col1.button("Predict Sentiment", type="primary")

if predict_clicked:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        result = predict_sentiment(user_text, model, tokenizer, label_map, config)
        emoji, color = SENTIMENT_STYLE.get(result["sentiment"], ("⚪", "#95a5a6"))

        st.markdown(
            f"### {emoji} Predicted sentiment: "
            f"<span style='color:{color}'>{result['sentiment']}</span> "
            f"({result['confidence']:.1%} confidence)",
            unsafe_allow_html=True,
        )

        with st.expander("Preprocessing & probability breakdown"):
            st.write("**Cleaned text fed to the model:**")
            st.code(result["cleaned"] or "(empty after cleaning)")

            prob_df = pd.DataFrame(
                {"Sentiment": list(result["probabilities"].keys()),
                 "Probability": list(result["probabilities"].values())}
            ).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Sentiment"))

st.divider()
st.caption(
    "Model, tokenizer, and label map are loaded exactly as saved in the notebook — "
    "predictions here should match the notebook's `predict_sentiment()` output for "
    "the same input."
)
