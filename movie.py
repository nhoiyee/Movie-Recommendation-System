import pandas as pd

# Define column names as per MovieLens 100k structure
columns = [
    "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# Load the u.item file (it's pipe-separated and encoded in Latin-1)
movies = pd.read_csv("ml-100k/u.item", sep='|', encoding='latin-1', names=columns)

# Extract genre names where value == 1
genre_cols = columns[5:]
movies['genres'] = movies[genre_cols].apply(lambda row: '|'.join([genre for genre, v in row.items() if v == 1]), axis=1)

# Keep only needed columns
movies_df = movies[['movie_id', 'title', 'genres']]

# Save to CSV
movies_df.to_csv("movies.csv", index=False)

print("✅ movies.csv created successfully.")
