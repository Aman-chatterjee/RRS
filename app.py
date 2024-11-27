from flask import Flask, request, jsonify, render_template
from recommender import RestaurantRecommender  # Import from the new module
import os

# Load the recommender model when the app starts
recommender = RestaurantRecommender.load_model('rr.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_preferences = request.get_json()
        user_preferences['average_cost'] = user_preferences['cost_for_one'] * 2
        recommendations = recommender.recommend(user_preferences)

        if isinstance(recommendations, str):
            return jsonify({"error": recommendations}), 400
        return jsonify(recommendations.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
