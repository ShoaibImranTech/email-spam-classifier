"""
Email Spam Predictor — Standalone Script
========================================
Trains a Naive Bayes model on combined_data.csv
and lets you predict any email as Spam or Ham.

Usage:
    python predict.py
"""

import re
import string
import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_data.csv')


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_train(model_name: str = 'svm'):
    """Load dataset, train chosen model, return (model, vectorizer)."""
    if not os.path.exists(DATA_PATH):
        print(f"❌  Dataset not found at: {DATA_PATH}")
        print("    Please place combined_data.csv inside the data/ folder.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df['clean'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v  = tfidf.transform(X_test)

    models = {
        'nb':  MultinomialNB(),
        'lr':  LogisticRegression(max_iter=1000, random_state=42),
        'svm': LinearSVC(C=1.0, random_state=42, max_iter=2000),
    }

    model = models.get(model_name, models['svm'])
    model.fit(X_train_v, y_train)
    preds = model.predict(X_test_v)

    print(f"\n✅ Model ({model_name.upper()}) trained.")
    print(f"   Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"   F1 Score : {f1_score(y_test, preds):.4f}\n")

    return model, tfidf


def predict_email(text: str, model, vectorizer) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return '🚫 SPAM' if pred == 1 else '✅ HAM (Not Spam)'


if __name__ == '__main__':
    print("=" * 50)
    print("  📧 Email Spam Classifier — Predictor")
    print("=" * 50)

    clf, vec = load_and_train(model_name='svm')

    test_emails = [
        "Congratulations! You've won a $1000 gift card. Click here to claim NOW!",
        "Hey, are we still on for the meeting tomorrow at 3pm?",
        "URGENT: Your account has been compromised. Verify immediately at http://scam.com",
        "Can you share the homework assignment? I missed the lecture today.",
    ]

    print("--- Sample Predictions ---")
    for email in test_emails:
        result = predict_email(email, clf, vec)
        print(f"  {result}")
        print(f"  Email: {email[:70]}...")
        print()

    print("--- Enter your own email (type 'quit' to exit) ---")
    while True:
        user_input = input("Email text: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if user_input:
            print(f"  → {predict_email(user_input, clf, vec)}\n")
