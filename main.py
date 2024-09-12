import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

# تحميل البيانات
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# إنشاء جدول بيانات يحتوي على تقييمات المستخدمين للأفلام
user_movie_ratings = ratings.pivot(
    index='userId', columns='movieId', values='rating').fillna(0)

# حساب التشابه بين المستخدمين
user_similarity = cosine_similarity(user_movie_ratings)
user_similarity_df = pd.DataFrame(
    user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

# حساب التشابه بين الأفلام
item_similarity = cosine_similarity(user_movie_ratings.T)
item_similarity_df = pd.DataFrame(
    item_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# دالة للتوصيات العامة بناءً على تقييمات المستخدمين


def general_recommendations(num_recommendations=5):
    mean_ratings = user_movie_ratings.mean().sort_values(ascending=False)
    return mean_ratings.head(num_recommendations)

# دالة للتوصيات بناءً على الأفلام المشاهدة


def recommend_similar_movies(movie_id, num_recommendations=5):
    similar_movies = item_similarity_df[movie_id].sort_values(
        ascending=False).head(num_recommendations)
    return similar_movies


# عرض التوصيات العامة
print("General Recommendataions")
print(general_recommendations())

# عرض توصيات بناءً على فيلم معين
print("Recommendations based on movie 1")
print(recommend_similar_movies(1))

# تقييم النموذج باستخدام RMSE


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# اختبار RMSE
test_ratings = [3, 5, 4]
predicted_ratings = [3.5, 4.8, 4.2]
print("RMSE:", rmse(test_ratings, predicted_ratings))
