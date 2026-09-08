"""
Trains a supervised learning model (Random Forest classifier) that predicts
whether a user will LIKE a movie (rating >= 4.0) based on:
  - the user's average rating & number of ratings given
  - the movie's average rating & number of ratings received
  - the movie's genres (one-hot encoded)

Run with:
    python src/train_model.py
"""

import os
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from features import build_training_table, load_data  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    print("Loading data...")
    movies, ratings = load_data(DATA_DIR)

    print("Building feature table...")
    X, y, feature_columns, genre_list, user_stats, movie_stats, movies_g = build_training_table(movies, ratings)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training RandomForestClassifier on {len(X_train)} ratings...")
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["not_liked", "liked"]))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    joblib.dump(genre_list, os.path.join(MODELS_DIR, "genre_list.pkl"))
    joblib.dump(user_stats, os.path.join(MODELS_DIR, "user_stats.pkl"))
    joblib.dump(movie_stats, os.path.join(MODELS_DIR, "movie_stats.pkl"))
    joblib.dump(movies_g, os.path.join(MODELS_DIR, "movies_g.pkl"))
    joblib.dump(ratings[["userId", "movieId"]], os.path.join(MODELS_DIR, "user_seen_movies.pkl"))

    print(f"Saved model + artifacts to '{MODELS_DIR}'")


if __name__ == "__main__":
    main()
