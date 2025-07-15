import pandas as pd
from scipy.sparse import coo_matrix, lil_matrix
from implicit.als import AlternatingLeastSquares
import os
import pickle

# Define base directory (no need to use os.chdir())
base_dir = "ml-100k"

# Load ratings data
ratings = pd.read_csv(os.path.join(base_dir, 'u.data'), sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load movie titles
movies = pd.read_csv(os.path.join(base_dir, "u.item"), sep="|", encoding="latin-1", names=[
    "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
])

# Sort user and item IDs
user_ids = sorted(ratings['user_id'].unique())
item_ids = sorted(ratings['item_id'].unique())

# Create mappings
user_id_mapping = {id_: idx for idx, id_ in enumerate(user_ids)}
item_id_mapping = {id_: idx for idx, id_ in enumerate(item_ids)}
reverse_item_mapping = {v: k for k, v in item_id_mapping.items()}

# Map original IDs to indices
ratings['user_idx'] = ratings['user_id'].map(user_id_mapping)
ratings['item_idx'] = ratings['item_id'].map(item_id_mapping)

# Create user-item matrix
n_users = len(user_id_mapping)
n_items = len(item_id_mapping)
user_item_matrix = coo_matrix(
    (ratings['rating'], (ratings['user_idx'], ratings['item_idx'])),
    shape=(n_users, n_items)
).tocsr()

# Transpose for training
item_user_matrix = user_item_matrix.T.tocsr()

# Train model
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=15)
# model.fit(item_user_matrix) 
model.fit(user_item_matrix)

# Generate recommendations for a test user
user_id = 5
if user_id not in user_id_mapping:
    raise ValueError(f"User ID {user_id} not found")

user_idx = user_id_mapping[user_id]

# Get recommendations
recommended = model.recommend(
    userid=user_idx,
    user_items=user_item_matrix[user_idx],
    N=10
)

# Movie lookup
movie_dict = pd.Series(movies.title.values, index=movies.movie_id).to_dict()

# Display recommendations
print("\nTop Recommended Movies:")
item_indices, scores = recommended
for item_idx, score in zip(item_indices, scores):
    movie_id = reverse_item_mapping.get(item_idx, "Unknown ID")
    title = movie_dict.get(movie_id, "Unknown Movie")
    print(f"{title} (ID: {movie_id}) - Score: {score:.2f}")

# Save the model
with open("als_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save mappings and data
with open("model_data.pkl", "wb") as f:
    pickle.dump({
        "user_id_mapping": user_id_mapping,
        "item_id_mapping": item_id_mapping,
        "reverse_item_mapping": reverse_item_mapping,
        "user_item_matrix": user_item_matrix,
        "movie_dict": movie_dict
    }, f)

import pickle

with open("als_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("mappings.pkl", "wb") as f:
    pickle.dump((
        user_item_matrix,
        user_id_mapping,
        item_id_mapping,
        reverse_item_mapping,
        movie_dict
    ), f)
    