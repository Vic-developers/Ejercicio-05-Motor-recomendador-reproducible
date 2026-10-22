from inf8239_u03.recommenders import ContentRecommender, MatrixFactorization, recommend_for_user


def test_cold_user_uses_global_popularity(sample_ratings, sample_movies):
    model = MatrixFactorization(factors=2, seed=7).fit(sample_ratings, epochs=1)
    content = ContentRecommender().fit(sample_movies)
    result = recommend_for_user(999, sample_ratings, sample_movies, model, content, k=3)
    assert len(result) == 3
    assert result['recommendation_source'].eq('global_popularity_cold_user').all()
    assert result['movieId'].is_unique


def test_new_user_with_history_uses_content_fallback(sample_ratings, sample_movies):
    model = MatrixFactorization(factors=2, seed=7).fit(sample_ratings, epochs=1)
    content = ContentRecommender().fit(sample_movies)
    history = sample_ratings[sample_ratings.userId.eq(1)].head(2).copy()
    history['userId'] = 999
    result = recommend_for_user(999, history, sample_movies, model, content, k=3)
    assert len(result) == 3
    assert result['recommendation_source'].eq('content_cold_user_with_history').all()
    assert not set(history.movieId) & set(result.movieId)


def test_known_user_uses_hybrid(sample_ratings, sample_movies):
    model = MatrixFactorization(factors=2, seed=7).fit(sample_ratings, epochs=1)
    content = ContentRecommender().fit(sample_movies)
    result = recommend_for_user(1, sample_ratings, sample_movies, model, content, alpha=0.75, k=2)
    assert len(result) == 2
    assert result['recommendation_source'].eq('hybrid').all()
    assert not set(sample_ratings.loc[sample_ratings.userId.eq(1), 'movieId']) & set(result.movieId)
