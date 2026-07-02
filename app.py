import os
import re
import string
import joblib
import streamlit as st

st.write("Current Directory:", os.getcwd())
st.write("Files:", os.listdir())

MODEL_PATH = r"C:\Users\saive\Downloads\streamlit_fake_news_app_files\fake_news_pipeline.pkl"


# IMPORTANT:
# This function must be available when loading the saved sklearn pipeline,
# because the notebook used it inside TfidfVectorizer(preprocessor=clean_text).
def clean_text(text):
    """Clean input news text before TF-IDF vectorization."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # remove URLs
    text = re.sub(r"<.*?>", " ", text)               # remove HTML tags
    text = re.sub(r"\d+", " ", text)                # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    """Load trained fake news pipeline."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found: fake_news_pipeline.pkl\n\n"
            "First run your notebook or train_model.py to create the model file."
        )
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


def prediction_label(value):
    """Convert model output into readable label."""
    text_value = str(value).strip().lower()

    # Handles common dataset labels
    if text_value in ["1", "fake", "false"]:
        return "Fake News"
    if text_value in ["0", "real", "true"]:
        return "Real News"

    return str(value)


st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline to predict whether it is Real or Fake.")

model = load_model()

news_text = st.text_area(
    "Paste news text here:",
    height=220,
    placeholder="Example: Government announces new economic policy today..."
)

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("Predict", use_container_width=True)

with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.rerun()

if predict_button:
    if model is None:
        st.stop()

    if not news_text.strip():
        st.warning("Please enter news text before prediction.")
        st.stop()

    try:
        pred = model.predict([news_text])[0]
        result = prediction_label(pred)

        st.subheader("Prediction Result")
        if result == "Fake News":
            st.error(f"🚨 {result}")
        elif result == "Real News":
            st.success(f"✅ {result}")
        else:
            st.info(f"Prediction: {result}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([news_text])[0]
            classes = model.classes_

            st.subheader("Prediction Confidence")
            confidence_data = {
                prediction_label(cls): round(float(prob) * 100, 2)
                for cls, prob in zip(classes, probabilities)
            }
            st.bar_chart(confidence_data)

            max_prob = max(probabilities) * 100
            st.write(f"Highest confidence: **{max_prob:.2f}%**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built using Streamlit, TF-IDF, and Machine Learning.")
