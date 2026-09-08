"""
Shared feature-engineering helpers for the movie recommendation system.

Both the training script and the Streamlit app use these functions so that
features are built the exact same way at train time and at prediction time.
"""

import pandas as pd

LIKE_THRESHOLD = 4.0  # a rating >= this value counts as "liked"


def load_data(data_dir="data"):
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    return movies, ratings


def get_genre_list(movies):
    """Return a sorted list of every individual genre in the dataset."""
    genres = set()
    for genre_str in movies["genres"]:
        if genre_str == "(no genres listed)":
            continue
        genres.update(genre_str.split("|"))
    return sorted(genres)


def add_genre_columns(movies, genre_list):
    """Add one boolean column per genre to the movies dataframe."""
    movies = movies.copy()
    for genre in genre_list:
        movies[f"genre_{genre}"] = movies["genres"].str.contains(genre, regex=False).astype(int)
    return movies


def compute_user_stats(ratings):
    stats = ratings.groupby("userId")["rating"].agg(user_avg_rating="mean", user_num_ratings="count")
    return stats.reset_index()


def compute_movie_stats(ratings):
    stats = ratings.groupby("movieId")["rating"].agg(movie_avg_rating="mean", movie_num_ratings="count")
    return stats.reset_index()


def build_feature_columns(genre_list):
    return ["user_avg_rating", "user_num_ratings", "movie_avg_rating", "movie_num_ratings"] + [
        f"genre_{g}" for g in genre_list
    ]


def build_training_table(movies, ratings):
    """Build the full (X, y) supervised-learning table from raw csv data."""
    genre_list = get_genre_list(movies)
    movies_g = add_genre_columns(movies, genre_list)

    user_stats = compute_user_stats(ratings)
    movie_stats = compute_movie_stats(ratings)

    df = ratings.merge(user_stats, on="userId").merge(movie_stats, on="movieId").merge(movies_g, on="movieId")

    # Remove the influence of the rating itself from the movie's average so the
    # model isn't trivially "cheating" by re-deriving the label out of the mean.
    df["movie_avg_rating"] = (df["movie_avg_rating"] * df["movie_num_ratings"] - df["rating"]) / (
        df["movie_num_ratings"] - 1
    ).replace(0, 1)

    df["liked"] = (df["rating"] >= LIKE_THRESHOLD).astype(int)

    feature_columns = build_feature_columns(genre_list)
    X = df[feature_columns]
    y = df["liked"]
    return X, y, feature_columns, genre_list, user_stats, movie_stats, movies_g
