import re
import string
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score


DATA_PATH = r"C:\Users\saive\Downloads\archive (7)\fakenews.csv"

MODEL_PATH =MODEL_PATH = r"C:\Users\saive\Downloads\streamlit_fake_news_app_files\fake_news_pipeline.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)

    text_col = "text"
    target_col = "label"

    df = df.drop_duplicates()
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.strip() != ""]
    df = df.dropna(subset=[target_col])

    X = df[text_col]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            stop_words="english"
        )),
        ("model", MultinomialNB())
    ])

    param_grid = {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__alpha": [0.1, 0.5, 1.0]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    main()
