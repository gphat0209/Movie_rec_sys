import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from flask import Flask, jsonify, render_template, request, session
# from fuzzywuzzy import process


app = Flask(__name__)
movie_data_df = pd.read_csv('data/preprocessed_data.csv')
user_rating_matrix = pd.read_csv("data/user_rating.csv", index_col=0)
n_recs = 5
movie_names = movie_data_df['title'].to_list()

cf_knn_model= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

def movie_recommender_engine(movie_data_df, movie_name, matrix, cf_model, n_recs):
    cf_model.fit(matrix)

    if movie_name not in movie_names:
        return False
    else: 
        movie_id = movie_data_df.loc[movie_data_df['title'] == movie_name, 'movieId'].iloc[0]

        # Calculate neighbors distance
        distances, indices = cf_model.kneighbors(matrix.loc[movie_id].values.reshape(1, -1), n_neighbors=n_recs)
        movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]

        # Get the result
        cf_recs = []
        for i in movie_rec_ids:
            id = matrix.index[i[0]]
            movie_title = movie_data_df.loc[movie_data_df['movieId'] == id, 'title'].iloc[0]
            cf_recs.append({'Title': movie_title,'Distance':i[1]})


        cf_recs_sorted = sorted(cf_recs, key=lambda x: x['Distance']) # Sort the result
        # df = pd.DataFrame(cf_recs_sorted, index = range(1,n_recs))
        movie_titles = [rec["Title"] for rec in cf_recs_sorted]
        return movie_titles



@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json.get('user_input')

        if user_input:
            recommendation = movie_recommender_engine(movie_data_df, user_input, user_rating_matrix, cf_knn_model, n_recs)
            if recommendation == False:
                return jsonify({"error": "Movie name isn't in database"})
            else:
                return jsonify({"recommended_movies": recommendation})
        else:
            return jsonify({"error": "No user input provided."}), 400
    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)