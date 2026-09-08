"""
Simple Streamlit UI for the Movie Recommendation System.

Run with:
    streamlit run app.py
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    genre_list = joblib.load(os.path.join(MODELS_DIR, "genre_list.pkl"))
    user_stats = joblib.load(os.path.join(MODELS_DIR, "user_stats.pkl"))
    movie_stats = joblib.load(os.path.join(MODELS_DIR, "movie_stats.pkl"))
    movies_g = joblib.load(os.path.join(MODELS_DIR, "movies_g.pkl"))
    seen = joblib.load(os.path.join(MODELS_DIR, "user_seen_movies.pkl"))
    return model, feature_columns, genre_list, user_stats, movie_stats, movies_g, seen


def recommend_for_user(user_id, top_n, model, feature_columns, user_stats, movie_stats, movies_g, seen):
    # Look up (or default) this user's average rating behaviour.
    row = user_stats[user_stats["userId"] == user_id]
    if len(row):
        user_avg_rating = row["user_avg_rating"].iloc[0]
        user_num_ratings = row["user_num_ratings"].iloc[0]
    else:
        user_avg_rating = user_stats["user_avg_rating"].mean()
        user_num_ratings = 0

    already_seen = set(seen.loc[seen["userId"] == user_id, "movieId"])
    candidates = movies_g[~movies_g["movieId"].isin(already_seen)].merge(movie_stats, on="movieId", how="left")
    candidates["movie_avg_rating"] = candidates["movie_avg_rating"].fillna(movie_stats["movie_avg_rating"].mean())
    candidates["movie_num_ratings"] = candidates["movie_num_ratings"].fillna(0)
    candidates["user_avg_rating"] = user_avg_rating
    candidates["user_num_ratings"] = user_num_ratings

    X = candidates[feature_columns]
    candidates["like_probability"] = model.predict_proba(X)[:, 1]

    top = candidates.sort_values("like_probability", ascending=False).head(top_n)
    return top[["title", "genres", "like_probability", "movie_avg_rating", "movie_num_ratings"]]


def main():
    st.title("🎬 Movie Recommendation System")
    st.caption("Supervised learning (Random Forest) trained on the MovieLens dataset.")

    if not os.path.exists(os.path.join(MODELS_DIR, "model.pkl")):
        st.error("No trained model found. Please run `python src/train_model.py` first.")
        return

    model, feature_columns, genre_list, user_stats, movie_stats, movies_g, seen = load_artifacts()

    user_ids = sorted(user_stats["userId"].unique())
    user_id = st.selectbox("Select a user", user_ids)
    top_n = st.slider("Number of recommendations", min_value=5, max_value=25, value=10)

    if st.button("Get Recommendations"):
        results = recommend_for_user(
            user_id, top_n, model, feature_columns, user_stats, movie_stats, movies_g, seen
        )
        st.subheader(f"Top {top_n} movies for user {user_id}")
        st.dataframe(
            results.rename(
                columns={
                    "title": "Title",
                    "genres": "Genres",
                    "like_probability": "Predicted like probability",
                    "movie_avg_rating": "Avg. rating",
                    "movie_num_ratings": "# ratings",
                }
            ).reset_index(drop=True),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
