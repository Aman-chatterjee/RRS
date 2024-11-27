import pandas as pd
import numpy as np
import pickle
import os
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify, render_template



class RestaurantRecommender:
    def __init__(self, restaurant_data):
        # Load restaurant data
        self.restaurants = pd.read_csv(restaurant_data, encoding='latin1')
        # Normalize city and cuisine case
        self.restaurants['City'] = self.restaurants['City'].str.lower()
        self.restaurants['Cuisines'] = self.restaurants['Cuisines'].str.lower().str.split(', ')

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the distance between two coordinates using Haversine formula."""
        R = 6371  # Earth radius in kilometers
        d_lat = radians(lat2 - lat1)
        d_lon = radians(lon2 - lon1)
        a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    def recommend(self, user_preferences):
        """
        Given user preferences, recommend the top 3 restaurants.
        """
        # Step 1: Filter restaurants for the same city
        filtered_restaurants = self.restaurants[self.restaurants['City'] == user_preferences['city']]
        if filtered_restaurants.empty:
            return "No restaurants found in the specified city."

        # Step 2: Calculate distance for restaurants in the same city
        filtered_restaurants['distance'] = filtered_restaurants.apply(
            lambda row: self.calculate_distance(user_preferences['latitude'], user_preferences['longitude'], 
                                                row['Latitude'], row['Longitude']), axis=1
        )
        filtered_restaurants = filtered_restaurants[filtered_restaurants['distance'] <= user_preferences['max_distance']]
        if filtered_restaurants.empty:
            return "No restaurants found within the specified maximum distance."

        # Step 3: Apply cuisine weights
        all_cuisines = set(cuisine for cuisines in filtered_restaurants['Cuisines'] for cuisine in cuisines)
        for cuisine in all_cuisines:
            weight = 2 if cuisine in user_preferences['cuisines'] else 1
            filtered_restaurants[cuisine] = filtered_restaurants['Cuisines'].apply(lambda x: weight if cuisine in x else 0)

        # Increase the weight of distance (e.g., multiply by 2)
        filtered_restaurants['distance_weighted'] = filtered_restaurants['distance'] * 2

        # Normalize numerical data with the weighted distance
        scaler = StandardScaler()
        numerical_data = scaler.fit_transform(
            filtered_restaurants[['Average Cost for two', 'Aggregate rating', 'distance_weighted']]
        )

        # Combine numerical data with one-hot encoded cuisines
        final_data = np.hstack((
            filtered_restaurants[list(all_cuisines)].values,  # Weighted cuisines
            numerical_data
        ))

        # Encode user preferences with weighted cuisines
        user_cuisines = [2 if cuisine in user_preferences['cuisines'] else 1 for cuisine in all_cuisines]

        # Normalize numerical preferences
        user_numerical = scaler.transform([[user_preferences['average_cost'], 
                                            user_preferences['min_rating'], 
                                            user_preferences['max_distance'] * 3]])

        # Combine user preferences into a single feature vector
        user_vector = np.hstack((user_cuisines, user_numerical.flatten()))

        # Train KNN Model
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(final_data)

        # Make Recommendations
        distances, indices = knn.kneighbors([user_vector])

        # Retrieve the recommended restaurants
        recommended_restaurants = filtered_restaurants.iloc[indices[0]]

        # Sort recommendations by distance or another column (if needed)
        recommended_restaurants = recommended_restaurants.sort_values(by='distance')

        # Show Top 3 Recommendations
        return recommended_restaurants[['Restaurant Name', 'City', 'Cuisines', 'Average Cost for two', 'Aggregate rating', 'distance']].head(3)


    def save_model(self, filename):
        """
        Save the recommender model (including KNN, StandardScaler, and restaurant data) to a pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved as {filename}")


    @staticmethod
    def load_model(filename):
        """
        Load the recommender model from a pickle file.
        """
        with open(filename, 'rb') as f:
            recommender = pickle.load(f)
        print(f"Model loaded from {filename}")
        return recommender


# Load the recommender model when the app starts
recommender = RestaurantRecommender.load_model('restaurant_recommender.pkl')


# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Define a route to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get user preferences from the request JSON
        user_preferences = request.get_json()

        # Ensure cost_for_one is converted to average_cost for two
        user_preferences['average_cost'] = user_preferences['cost_for_one'] * 2
        
        # Get recommendations from the model
        recommendations = recommender.recommend(user_preferences)
        
        if isinstance(recommendations, str):
            return jsonify({"error": recommendations}), 400  # Error message if no recommendations
        
        # Return recommendations as JSON
        return jsonify(recommendations.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error if something goes wrong

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

