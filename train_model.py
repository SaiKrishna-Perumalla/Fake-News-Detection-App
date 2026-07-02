import re
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Adjust these column names to match your dataset
X = df["text"].apply(clean_text)
y = df["label"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "fake_news_pipeline.pkl")

print("Model saved successfully!")
