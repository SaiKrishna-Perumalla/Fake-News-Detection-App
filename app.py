import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ---------------------------
# Locate Model
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_pipeline.pkl")

# Debug Information (Remove after deployment if desired)
with st.expander("Debug Information"):
    st.write("Current Directory:", BASE_DIR)
    st.write("Files:", os.listdir(BASE_DIR))
    st.write("Model Path:", MODEL_PATH)
    st.write("Model Exists:", os.path.exists(MODEL_PATH))

# ---------------------------
# Load Model
# ---------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found: fake_news_pipeline.pkl")
    st.info("Run train_model.py first to generate the model.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model:\n\n{e}")
    st.stop()

# ---------------------------
# Title
# ---------------------------
st.title("📰 Fake News Detection App")

st.write(
    "Enter a news headline or article below to determine whether it is **Real** or **Fake**."
)

# ---------------------------
# Text Input
# ---------------------------
news = st.text_area(
    "Paste News Text",
    height=250,
    placeholder="Example: Government announces new economic policy..."
)

col1, col2 = st.columns(2)

predict = col1.button("Predict", use_container_width=True)
clear = col2.button("Clear", use_container_width=True)

if clear:
    st.rerun()

# ---------------------------
# Prediction
# ---------------------------
if predict:

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        try:
            prediction = model.predict([news])[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba([news])[0]
                confidence = np.max(probability) * 100
            else:
                confidence = None

            st.markdown("## Prediction Result")

            if prediction == 1:
                st.success("✅ Real News")
            else:
                st.error("❌ Fake News")

            if confidence is not None:

                st.markdown("## Prediction Confidence")

                df = pd.DataFrame({
                    "Prediction": ["Confidence"],
                    "Score": [confidence]
                })

                st.bar_chart(
                    df.set_index("Prediction")
                )

                st.write(f"Confidence: **{confidence:.2f}%**")

        except Exception as e:
            st.error(f"Prediction Error:\n\n{e}")
