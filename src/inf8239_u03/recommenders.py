from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def weighted_popularity(ratings: pd.DataFrame, movies: pd.DataFrame, quantile: float = 0.80) -> pd.DataFrame:
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    global_mean = float(ratings["rating"].mean())
    minimum = float(stats["count"].quantile(quantile))
    stats["weighted_score"] = (
        stats["count"] / (stats["count"] + minimum) * stats["mean"]
        + minimum / (stats["count"] + minimum) * global_mean
    )
    return stats.query("count >= @minimum").join(movies.set_index("movieId")).sort_values("weighted_score", ascending=False)


class ContentRecommender:
    def fit(self, movies: pd.DataFrame) -> "ContentRecommender":
        self.movies = movies.reset_index(drop=True).copy()
        self.movies["genres_text"] = self.movies["genres"].str.replace("|", " ", regex=False)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.movies["genres_text"])
        return self

    def recommend(self, title: str, k: int = 10) -> pd.DataFrame:
        matches = self.movies.index[self.movies["title"].eq(title)]
        if len(matches) == 0:
            raise KeyError(f"Título no encontrado: {title}")
        index = int(matches[0])
        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        order = [value for value in scores.argsort()[::-1] if value != index][:k]
        result = self.movies.loc[order, ["movieId", "title", "genres"]].copy()
        result["content_score"] = scores[order]
        return result.reset_index(drop=True)


class MatrixFactorization:
    def __init__(self, factors: int = 20, learning_rate: float = 0.01, regularization: float = 0.05, seed: int = 42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed

    def fit(self, ratings: pd.DataFrame, epochs: int = 12) -> "MatrixFactorization":
        self.users = sorted(ratings["userId"].unique())
        self.items = sorted(ratings["movieId"].unique())
        self.user_index = {value: index for index, value in enumerate(self.users)}
        self.item_index = {value: index for index, value in enumerate(self.items)}
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (len(self.users), self.factors))
        self.item_factors = rng.normal(0, 0.1, (len(self.items), self.factors))
        self.global_mean = float(ratings["rating"].mean())
        observations = [(self.user_index[r.userId], self.item_index[r.movieId], float(r.rating)) for r in ratings.itertuples()]
        for _ in range(epochs):
            rng.shuffle(observations)
            for user, item, rating in observations:
                error = rating - self.predict_indices(user, item)
                previous_user = self.user_factors[user].copy()
                self.user_factors[user] += self.learning_rate * (error * self.item_factors[item] - self.regularization * self.user_factors[user])
                self.item_factors[item] += self.learning_rate * (error * previous_user - self.regularization * self.item_factors[item])
        return self

    def predict_indices(self, user_index: int, item_index: int) -> float:
        return float(self.global_mean + self.user_factors[user_index] @ self.item_factors[item_index])

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.predict_indices(self.user_index[user_id], self.item_index[movie_id])

    def top_n(self, user_id: int, seen: set[int], k: int = 10) -> pd.DataFrame:
        if user_id not in self.user_index:
            return pd.DataFrame(columns=["movieId", "collaborative_score"])
        candidates = [item for item in self.items if item not in seen]
        item_indices = np.asarray([self.item_index[item] for item in candidates], dtype=int)
        user_vector = self.user_factors[self.user_index[user_id]]
        scores = self.global_mean + self.item_factors[item_indices] @ user_vector
        order = scores.argsort()[::-1][:k]
        return pd.DataFrame({"movieId": np.asarray(candidates)[order], "collaborative_score": scores[order]})


def temporal_leave_one_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"], kind="mergesort")
    test = ordered.groupby("userId", sort=False).tail(1)
    train = ordered.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    collaborative_model: MatrixFactorization,
    content_model: ContentRecommender,
    alpha: float = 0.75,
    k: int = 10,
    candidate_pool: int = 100,
) -> pd.DataFrame:
    """Return a recommendation list with an explicit no-history/user-cold fallback."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe estar entre 0 y 1")
    if k < 1 or candidate_pool < 1:
        raise ValueError("k y candidate_pool deben ser positivos")
    seen_rows = ratings.loc[ratings["userId"].eq(user_id), ["movieId", "rating"]]
    seen = set(seen_rows["movieId"].astype(int))
    if seen_rows.empty:
        popular = weighted_popularity(ratings, movies).reset_index()
        popular = popular.loc[~popular["movieId"].isin(seen)].sort_values(["weighted_score", "count", "movieId"], ascending=[False, False, True]).head(k)
        popular = popular.rename(columns={"weighted_score": "score"})
        popular["recommendation_source"] = "global_popularity_cold_user"
        return popular[["movieId", "title", "genres", "score", "recommendation_source"]].reset_index(drop=True)

    movie_rows = {int(movie_id): row for row, movie_id in enumerate(content_model.movies["movieId"])}
    history = seen_rows.loc[seen_rows["movieId"].isin(movie_rows)]
    history_rows = [movie_rows[int(movie_id)] for movie_id in history["movieId"]]
    weights = np.clip(history["rating"].to_numpy(dtype=float) - 2.5, 0.1, None)
    profile = content_model.matrix[history_rows].multiply(weights[:, None]).sum(axis=0) / weights.sum()
    profile = np.asarray(profile).reshape(1, -1)
    candidates = content_model.movies.loc[~content_model.movies["movieId"].isin(seen)].copy()
    candidate_rows = [movie_rows[int(movie_id)] for movie_id in candidates["movieId"]]
    content_scores = cosine_similarity(profile, content_model.matrix[candidate_rows]).ravel()
    candidates["content_score"] = content_scores

    if user_id not in collaborative_model.user_index:
        candidates["score"] = candidates["content_score"]
        candidates["recommendation_source"] = "content_cold_user_with_history"
    else:
        candidate_ids = set(collaborative_model.items)
        candidates = candidates.loc[candidates["movieId"].isin(candidate_ids)].copy()
        collaborative = collaborative_model.top_n(user_id, seen, max(candidate_pool, k))
        candidates = candidates.merge(collaborative, on="movieId", how="inner")
        collab = candidates["collaborative_score"].to_numpy()
        collab = (collab - collab.min()) / max(collab.max() - collab.min(), 1e-12)
        content = candidates["content_score"].to_numpy()
        content = (content - content.min()) / max(content.max() - content.min(), 1e-12)
        candidates["score"] = alpha * collab + (1.0 - alpha) * content
        candidates["recommendation_source"] = "hybrid"
    return candidates.nlargest(k, "score")[["movieId", "title", "genres", "score", "recommendation_source"]].reset_index(drop=True)
