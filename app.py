from flask import Flask, request, jsonify
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = Flask(__name__)
movies = pd.read_csv('tmdb_5000/tmdb_5000_movies.csv')

# Combine genres into a single string for each movie
movies['combined_genres'] = movies['genres'].apply(lambda x: ' '.join([d['name'] for d in ast.literal_eval(x)]))

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

emotion_to_genres = {
    "Angry": ["Action", "Thriller","Crime"],
    "Disgusted": ["Horror", "Mystery"],
    "Fearful": ["Horror", "Thriller"],
    "Happy": ["Comedy", "Adventure"],
    "Neutral": ["Drama", "Documentary"],
    "Sad": ["Drama", "Romance"],
    "Surprised": ["Fantasy", "ScienceFiction"]
}

def get_mood_based_recommendations(emotion, n_recommendations=10):
    genres = emotion_to_genres.get(emotion, [])
    if not genres:
        return []
    
    filtered_movies = movies[movies['combined_genres'].apply(lambda x: any(genre.lower() in x.lower() for genre in genres))]
    return filtered_movies['title'].head(n_recommendations).tolist()

def get_similar_movies(title, n_recommendations=10):
    idx = movies[movies['title'] == title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:n_recommendations+1]
    
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices].tolist()


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    data = request.json
    mood = data['mood']
    recommendations = get_mood_based_recommendations(mood)
    return jsonify(recommendations)

@app.route('/similar', methods=['GET', 'POST'])
def similar():
    data = request.json
    title = data['title']
    recommendations = get_similar_movies(title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)