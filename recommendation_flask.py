from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd


app = Flask(__name__)
CORS(app)


# === Load model and data ===
with open('als_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('mappings.pkl', 'rb') as f:
    user_item_matrix, user_id_mapping, item_id_mapping, reverse_item_mapping, movie_dict = pickle.load(f)

movies_df = pd.read_csv("movies.csv")

# === Recommendation Function ===
def get_recommendations(user_id, N=5):
    if user_id not in user_id_mapping:
        return []

    user_index = user_id_mapping[user_id]

    # model.recommend returns: (recommended_items, scores)
    item_indices, scores = model.recommend(user_index, user_item_matrix[user_index], N=N)

    results = []
    for item_index, score in zip(item_indices, scores):
        item_index = int(item_index)  # convert np.float32 to int
        movie_id = reverse_item_mapping.get(item_index)
        title = movie_dict.get(movie_id, "Unknown")
        results.append({"title": title, "score": float(score)})

    return results

# === API Endpoint ===
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    recommendations = get_recommendations(user_id)
    return jsonify(recommendations)


# recommend_by_title Endpoint takes a movie_id, and returns similar items.

@app.route('/recommend_by_title', methods=['GET'])
def recommend_by_title():
    movie_id = request.args.get('movie_id', type=int)
    top_n = request.args.get('N', default=5, type=int)

    if movie_id not in item_id_mapping:
        return jsonify({"error": "Movie ID not found"}), 404

    item_index = item_id_mapping[movie_id]
    recommended = model.similar_items(item_index, N=top_n + 1)  # +1 to exclude the movie itself

    results = []
    for idx, score in recommended:
        if idx == item_index:
            continue  # skip the movie itself
        movie_id_rec = reverse_item_mapping[idx]
        title = movie_dict.get(movie_id_rec, "Unknown")
        results.append({"title": title, "score": float(score)})

    return jsonify(results)

# recommend_by_genre Endpoint - Returns the most popular movies in a given genre
@app.route('/recommend_by_genre', methods=['GET'])
def recommend_by_genre():
    genre = request.args.get('genre', type=str)
    top_n = request.args.get('N', default=5, type=int)

    if not genre:
        return jsonify({"error": "Genre is required"}), 400

    # Filter movies with this genre
    genre_movies = movies_df[movies_df["genres"].str.contains(genre, na=False)]

    if genre_movies.empty:
        return jsonify([])

    # Pick top N by popularity (total user ratings)
    genre_ids = set(genre_movies["movie_id"])
    genre_items = [item_id_mapping[m] for m in genre_ids if m in item_id_mapping]

    item_scores = []
    for item in genre_items:
        total_score = user_item_matrix[:, item].sum()
        item_scores.append((item, total_score))

    # Sort by popularity
    item_scores = sorted(item_scores, key=lambda x: -x[1])[:top_n]

    results = []
    for item_index, score in item_scores:
        movie_id = reverse_item_mapping[item_index]
        title = movie_dict.get(movie_id, "Unknown")
        results.append({"title": title, "score": float(score)})

    return jsonify(results)

# === Run the API ===
if __name__ == '__main__':
    print("✅ Running Flask on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)