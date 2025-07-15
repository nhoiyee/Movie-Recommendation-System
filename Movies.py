#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Data Collection & Preprocessing

# In[8]:


import pandas as pd

# Load ratings and movies data


ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                     names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])


# ## Handle Missing Values

# In[9]:


# Check for missing values
print(ratings.isnull().sum())
print(movies.isnull().sum())

# Handle missing values if any
ratings.dropna(inplace=True)
movies.dropna(inplace=True)


# ## Handle Duplicates

# In[10]:


# Drop duplicate rows if they exist
ratings.drop_duplicates(inplace=True)
movies.drop_duplicates(inplace=True)


# ## Outlier Detection

# In[11]:


# Since ratings range from 1–5, filter any invalid ratings if present.

ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]


# ## Feature Extraction

# In[ ]:


# extract:

# From ratings: user_id, movie_id, rating, timestamp
    
# From movies: movie_id, title, genres


# In[12]:


# Combine genre columns into a single list per movie
genre_cols = movies.columns[5:]
movies['genres'] = movies[genre_cols].apply(lambda row: [genre for genre in genre_cols if row[genre] == 1], axis=1)

# Merge datasets on movie_id
movie_data = pd.merge(ratings, movies[['movie_id', 'title', 'genres']], on='movie_id')


# In[13]:


print(movie_data.head())


# ## Step 2: Exploratory Data Analysis (EDA)

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


# Set plot style
sns.set(style='whitegrid')

# Plot rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=ratings, palette='coolwarm')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# In[16]:


# Analyze User Interaction
# Ratings per user
user_counts = ratings['user_id'].value_counts()

plt.figure(figsize=(10, 5))
sns.histplot(user_counts, bins=50, kde=True)
plt.title('Number of Ratings per User')
plt.xlabel('Ratings Count')
plt.ylabel('Number of Users')
plt.show()


# In[17]:


# Analyze the distribution of ratings and user interactions.

# This shows which ratings are most common (usually 4s and 5s are dominant).
# This shows how active users are. Some users rate a lot; others rate just a few.


# In[ ]:


# check


# In[26]:


movies = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    usecols=[0, 1],
    names=['movie_id', 'title']
)


# In[27]:


print(movies.head())
print("Unique movie_ids in movies:", movies['movie_id'].nunique())


# In[29]:


movie_rating_counts = ratings['movie_id'].value_counts().reset_index()
movie_rating_counts.columns = ['movie_id', 'rating_count']

most_rated_movies = movie_rating_counts.merge(movies, on='movie_id', how='inner')

# Plot Top 10
top_10 = most_rated_movies.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y='title', x='rating_count', data=top_10, palette='magma')
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()



# In[30]:


# shows top 10 Most Rated Movies


# ## Show Genre Preferences

# In[31]:


# how many movies belong to each genre
# load the full genre data
genre_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies_full = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    names=genre_columns
)


# In[32]:


# Sum Each Genre Column
# Select genre columns only
genre_data = movies_full.iloc[:, 5:]  # Columns 5 onwards are genre flags

# Sum up each genre
genre_counts = genre_data.sum().sort_values(ascending=False)

# Print counts
print(genre_counts)


# In[33]:


# Plot Genre Preferences
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
plt.title('Genre Preferences (Number of Movies per Genre)')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# In[34]:


# Show Genre Preferences Based on Ratings
# Merge ratings with full movie genre data
ratings_genres = ratings.merge(movies_full[['movie_id'] + list(genre_data.columns)], on='movie_id')

# Multiply each rating by each genre flag to count ratings per genre
genre_ratings = ratings_genres[genre_data.columns].sum().sort_values(ascending=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_ratings.values, y=genre_ratings.index, palette='plasma')
plt.title('Genre Preferences (Based on Number of Ratings)')
plt.xlabel('Number of Ratings')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# In[35]:


# Identify patterns in user-movie interactions.


# ## Ratings per User (Activity Level)

# In[36]:


user_activity = ratings['user_id'].value_counts()

plt.figure(figsize=(10, 5))
sns.histplot(user_activity, bins=50, kde=True, color='skyblue')
plt.title('Distribution of Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()


# ## Ratings per Movie (Movie Popularity)

# In[37]:


# how frequently each movie is rated.

movie_popularity = ratings['movie_id'].value_counts()

plt.figure(figsize=(10, 5))
sns.histplot(movie_popularity, bins=50, kde=True, color='salmon')
plt.title('Distribution of Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()


# In[ ]:


# Sparsity of the User-Movie Matrix:
# If sparsity is > 90%, that means most user-movie pairs have no rating. (impt for  for collaborative filtering)


# In[38]:


num_users = ratings['user_id'].nunique()
num_movies = ratings['movie_id'].nunique()
num_interactions = len(ratings)

sparsity = 1.0 - (num_interactions / (num_users * num_movies))
print(f"User-Movie Matrix Sparsity: {sparsity:.2%}")


# ## Average Rating Per User

# In[40]:


avg_rating_per_user = ratings.groupby('user_id')['rating'].mean()

plt.figure(figsize=(10, 5))
sns.histplot(avg_rating_per_user, bins=40, kde=True, color='green')
plt.title('Average Rating per User')
plt.xlabel('Average Rating')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.show()


# In[42]:


# explore when users are active
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Count ratings over time
ratings_by_date = ratings['datetime'].dt.to_period('M').value_counts().sort_index()

plt.figure(figsize=(12, 5))
ratings_by_date.plot(kind='bar', color='purple')
plt.title('Number of Ratings per Month')
plt.xlabel('Month')
plt.ylabel('Number of Ratings')
plt.tight_layout()
plt.show()


# ## Step 3: Implement Content-Based Filtering

# In[43]:


# One-Hot Encode Genres
# Genres are already in binary columns (0 or 1), so use them as features.
# Select movie_id, title, and genre columns
genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movie_features = movies_full[['movie_id', 'title'] + genre_columns].copy()


# In[44]:


# Cosine similarity is used on genre vectors to 
# measure the similarity/ directional relationship between different genres, use Cosine Similarity with high-dimensional data
# ( number of features in dataset is very large)


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity matrix
genre_matrix = movie_features[genre_columns].values
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Store the index mapping for quick lookup
movie_indices = pd.Series(movie_features.index, index=movie_features['title'])


# ## Define Recommendation Function

# In[46]:


# Recommend similar movies based on a given movie title


# In[47]:


def recommend_similar_movies(title, top_n=5):
    if title not in movie_indices:
        return f"❌ Movie '{title}' not found in dataset."

    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the movie itself and get top N
    top_indices = [i for i, _ in sim_scores[1:top_n+1]]
    return movie_features.iloc[top_indices][['title']]


# In[48]:


recommend_similar_movies('Star Wars (1977)', top_n=5)


# In[50]:


recommend_similar_movies('Alien', top_n=5)


# In[55]:


recommend_similar_movies('Aliens (1986)', top_n=5)


# In[56]:


recommend_similar_movies('Terminator, The (1984)', top_n=5)


# ## Step 4: Implement Collaborative Filtering

# In[1]:


from surprise import SVD, Dataset, Reader
print("Surprise imported successfully ✅")


# In[3]:


from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import pandas as pd



# In[5]:


# Load Ratings Data

# columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Surprise expects only (user, item, rating)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)



# In[6]:


#Train-Test Split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

#Train an SVD Model
model = SVD()
model.fit(trainset)

#Evaluate Model
predictions = model.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)


# In[7]:


# generate top N recommendations for a user based on predicted ratings.

def get_top_n_recommendations(predictions, user_id, movie_df, n=10):
    # Filter predictions for this user
    user_preds = [pred for pred in predictions if pred.uid == str(user_id)]
    user_preds.sort(key=lambda x: x.est, reverse=True)

    top_n = user_preds[:n]
    top_movie_ids = [int(pred.iid) for pred in top_n]
    
    # Map movie IDs to titles
    return movie_df[movie_df['movie_id'].isin(top_movie_ids)][['movie_id', 'title']]

# Example usage
# Load movies
movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                        names=['movie_id', 'title'], usecols=[0, 1])
top_recs = get_top_n_recommendations(predictions, user_id=1, movie_df=movies_df)
print(top_recs)


# ## KNN-Based Collaborative Filtering

# In[8]:


from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy


# In[9]:


from surprise import Dataset, Reader

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# ## KNN User-Based Collaborative Filtering

# In[10]:


sim_options = {
    'name': 'cosine',
    'user_based': True  # User-based CF
}

user_knn = KNNBasic(sim_options=sim_options)
user_knn.fit(trainset)

user_predictions = user_knn.test(testset)
print("User-based KNN:")
accuracy.rmse(user_predictions)
accuracy.mae(user_predictions)


# ## KNN Item-Based Collaborative Filtering

# In[11]:


sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based CF
}

item_knn = KNNBasic(sim_options=sim_options)
item_knn.fit(trainset)

item_predictions = item_knn.test(testset)
print("Item-based KNN:")
accuracy.rmse(item_predictions)
accuracy.mae(item_predictions)


# In[12]:


# User-based CF may work well when users have consistent behavior patterns.

# Item-based CF can be more stable and scalable, especially when new users are sparse (cold start problem).


# ## Step 5: Model Evaluation

# In[13]:


# compare the performance of collaborative filtering (SVD, KNN) and content-based filtering models.

# SVD

# User-based KNN

# Item-based KNN


# In[15]:


# RMSE & MAE on Test Set

from surprise import accuracy

# SVD
svd_predictions = model.test(testset)
print("SVD:")
print("  RMSE:", accuracy.rmse(svd_predictions))
print("  MAE:", accuracy.mae(svd_predictions))

# User-based KNN
user_predictions = user_knn.test(testset)
print("User-based KNN:")
print("  RMSE:", accuracy.rmse(user_predictions))
print("  MAE:", accuracy.mae(user_predictions))

# Item-based KNN
item_predictions = item_knn.test(testset)
print("Item-based KNN:")
print("  RMSE:", accuracy.rmse(item_predictions))
print("  MAE:", accuracy.mae(item_predictions))


# In[17]:


# Cross-Validation (to assess generalization)
from surprise.model_selection import cross_validate

print("SVD Cross-Validation:")
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("User-based KNN Cross-Validation:")
cross_validate(user_knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("Item-based KNN Cross-Validation:")
cross_validate(item_knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[18]:


# Evaluate Content-Based Filtering
#  generate predictions (similarity scores), Evaluate using Precision@K, Recall@K, or Top-N hit rate


# In[19]:


from collections import defaultdict

def precision_recall_at_k(predictions, k=5, threshold=4.0):
    user_tp = defaultdict(int)
    user_fp = defaultdict(int)
    user_fn = defaultdict(int)

    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold and true_r >= threshold:
            user_tp[uid] += 1
        elif est >= threshold and true_r < threshold:
            user_fp[uid] += 1
        elif est < threshold and true_r >= threshold:
            user_fn[uid] += 1

    precisions = {uid: user_tp[uid] / (user_tp[uid] + user_fp[uid] + 1e-10) for uid in user_tp}
    recalls = {uid: user_tp[uid] / (user_tp[uid] + user_fn[uid] + 1e-10) for uid in user_tp}

    return sum(precisions.values()) / len(precisions), sum(recalls.values()) / len(recalls)

precision, recall = precision_recall_at_k(svd_predictions, k=5, threshold=4.0)
print(f"SVD Precision@5: {precision:.4f}, Recall@5: {recall:.4f}")


