from flask import Flask, request, jsonify, render_template
import os
from recommender import RestaurantRecommender  # Import the class from the file where it's defined


# Load the recommender model when the app starts
recommender = RestaurantRecommender.load_model('./restaurant_recommender.pkl')


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

