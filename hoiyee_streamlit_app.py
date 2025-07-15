import streamlit as st
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import os

# Load and preprocess data
@st.cache_data
def load_data():
    # os.chdir("./ml-100k/ml-100k")
    # ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    # movies = pd.read_csv("u.item", sep="|", encoding="latin-1", names=[
    #     "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    #     "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    #     "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    #     "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    # ])

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=[
    "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
])


    user_ids = sorted(ratings['user_id'].unique())
    item_ids = sorted(ratings['item_id'].unique())

    user_id_mapping = {id_: idx for idx, id_ in enumerate(user_ids)}
    item_id_mapping = {id_: idx for idx, id_ in enumerate(item_ids)}
    reverse_item_mapping = {v: k for k, v in item_id_mapping.items()}
    movie_dict = pd.Series(movies.title.values, index=movies.movie_id).to_dict()

    ratings['user_idx'] = ratings['user_id'].map(user_id_mapping)
    ratings['item_idx'] = ratings['item_id'].map(item_id_mapping)

    user_item_matrix = coo_matrix(
        (ratings['rating'], (ratings['user_idx'], ratings['item_idx'])),
        shape=(len(user_id_mapping), len(item_id_mapping))
    ).tocsr()

    item_user_matrix = user_item_matrix.T.tocsr()

    return user_item_matrix, item_user_matrix, user_id_mapping, reverse_item_mapping, movie_dict

# Load data
user_item_matrix, item_user_matrix, user_id_mapping, reverse_item_mapping, movie_dict = load_data()

# Train model
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)
model.fit(user_item_matrix)

# Streamlit interface
st.title("Movie Recommender (ALS Model)")

user_id = st.number_input("Enter User ID", min_value=1, max_value=max(user_id_mapping.keys()))

if int(user_id) in user_id_mapping:
    user_idx = user_id_mapping[int(user_id)]
    user_idx = int(user_idx)

    if user_idx >= user_item_matrix.shape[0]:
        st.error(f"User index {user_idx} is out of bounds.")
    else:
        if user_item_matrix[user_idx].nnz == 0:
            st.warning("This user has no ratings. Please try a different user.")
        else:
            recommended = model.recommend(
                userid=user_idx,
                user_items=user_item_matrix[user_idx],
                N=10
            )

            st.subheader("Top 10 Recommended Movies:")
            item_indices, scores = recommended
            for item_idx, score in zip(item_indices, scores):
                movie_id = reverse_item_mapping.get(item_idx, "Unknown ID")
                title = movie_dict.get(movie_id, "Unknown Movie")
                st.write(f"{title} (ID: {movie_id}) - Score: {score:.2f}")
else:
    st.error("User ID not found.")

