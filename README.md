# Movie Recommendation System

A simple movie recommendation system built with **Python** and a **supervised
learning** algorithm (Random Forest classifier), with a minimal **Streamlit**
UI.

## How it works

1. We use the [MovieLens (ml-latest-small)](https://grouplens.org/datasets/movielens/) dataset (`data/movies.csv`, `data/ratings.csv`).
2. For every (user, movie) rating we build features:
   - the user's average rating & number of ratings given
   - the movie's average rating & number of ratings received
   - the movie's genres (one-hot encoded)
3. The label is `liked = 1` if the rating is `>= 4.0`, else `0`.
4. A `RandomForestClassifier` is trained to predict `liked` from these features.
5. To recommend movies for a user, we predict the "like probability" for every
   movie they haven't rated yet, and show the top N.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python src/train_model.py
```

This prints the model's test accuracy/classification report and saves the
trained model + supporting data to `models/`.

## Run the UI

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501),
pick a user ID, and click **Get Recommendations**.

## Project structure

```
data/                MovieLens csv files (movies.csv, ratings.csv, ...)
models/              Saved model + feature artifacts (created by train_model.py)
src/
  features.py         Shared feature-engineering code
  train_model.py       Trains and saves the Random Forest model
app.py                Streamlit UI
requirements.txt
```
