import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

class RestaurantRecommender:
    def __init__(self, restaurant_data):
        self.restaurants = pd.read_csv(restaurant_data, encoding='latin1')
        self.restaurants['City'] = self.restaurants['City'].str.lower()
        self.restaurants['Cuisines'] = self.restaurants['Cuisines'].str.lower().str.split(', ')

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        d_lat = radians(lat2 - lat1)
        d_lon = radians(lon2 - lon1)
        a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def recommend(self, user_preferences):
        # Same recommend logic...
        pass

    def save_model(self, filename):
        joblib.dump(self, filename)
        print(f"Model saved as {filename}")

    @staticmethod
    def load_model(filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
