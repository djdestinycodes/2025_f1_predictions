"""
Race Predictor Module
--------------------
This module contains the RacePredictor class that handles F1 race predictions using
machine learning. It uses historical race data, qualifying times, and weather data
to predict race outcomes.
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class RacePredictor:
    """
    A class to predict F1 race results using machine learning.
    
    This class uses historical race data, qualifying times, and weather data to train a model
    that can predict race outcomes. It takes into account various factors such as
    qualifying performance, historical race data, driver consistency, and weather conditions.
    """
    
    def __init__(self, cache_dir="data/f1_cache"):
        """
        Initialize the RacePredictor.
        
        Args:
            cache_dir (str): Directory to store FastF1 cached data
        """
        # Enable FastF1 caching
        fastf1.Cache.enable_cache(cache_dir)
        
        # Initialize model with optimized parameters
        self.model = GradientBoostingRegressor(
            n_estimators=800,
            learning_rate=0.015,
            max_depth=7,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.6,
            max_features='sqrt',
            random_state=42
        )
        
        # Define feature names for the model
        self.feature_names = [
            "QualifyingTime (s)",
            "QualifyingPosition",
            "QualifyingGapToLeader",
            "QualifyingConsistency",
            "TeamPerformance",
            "Temperature",
            "Humidity",
            "RainProbability",
            "WindSpeed"
        ]
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), self.feature_names)
            ])
        
        # Create full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
    
    def load_historical_data(self, year, race_number):
        """
        Load historical race data for training.
        
        Args:
            year (int): Year of the historical race
            race_number (int): Race number in the season
            
        Returns:
            pd.DataFrame: Historical race data with driver statistics and weather data
        """
        try:
            print(f"Loading historical data for race number {race_number}...")
            session = fastf1.get_session(year, race_number, "R")
            session.load()
            
            # Extract relevant features from lap data
            laps = session.laps[["Driver", "LapTime", "Position"]].copy()
            
            # Convert times to seconds
            laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
            
            # Calculate driver statistics
            driver_stats = laps.groupby("Driver").agg({
                "LapTime (s)": ["mean", "std"],
                "Position": "mean"
            }).reset_index()
            
            # Flatten column names
            driver_stats.columns = ['_'.join(col).strip('_') for col in driver_stats.columns.values]
            
            # Calculate consistency score (lower std = more consistent)
            driver_stats["ConsistencyScore"] = 1 / (1 + driver_stats["LapTime (s)_std"])
            
            # Rename columns for clarity
            driver_stats = driver_stats.rename(columns={
                "LapTime (s)_mean": "AvgLapTime (s)"
            })
            
            # Add weather data from FastF1
            weather_data = session.weather_data
            if weather_data is not None and not weather_data.empty:
                # Calculate average weather conditions during the race
                avg_weather = weather_data.mean()
                driver_stats["Temperature"] = avg_weather.get("AirTemp", 25.0)
                driver_stats["Humidity"] = avg_weather.get("Humidity", 50.0)
                driver_stats["RainProbability"] = avg_weather.get("Rainfall", 0.0)
                driver_stats["WindSpeed"] = avg_weather.get("WindSpeed", 5.0)
            else:
                # Set default weather values if no data available
                driver_stats["Temperature"] = 25.0  # Default temperature in Celsius
                driver_stats["Humidity"] = 50.0     # Default humidity percentage
                driver_stats["RainProbability"] = 0.0  # Default rain probability
                driver_stats["WindSpeed"] = 5.0     # Default wind speed in km/h
            
            print(f"Successfully loaded data for {len(driver_stats)} drivers")
            return driver_stats
            
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            # Return a default dataset with basic statistics and weather data
            return pd.DataFrame({
                "Driver": ["VER", "NOR", "PIA", "RUS", "TSU", "ALB", "LEC", "HAM", "GAS", "SAI", "ALO", "STR"],
                "AvgLapTime (s)": [90.0, 90.5, 91.0, 91.5, 92.0, 92.5, 93.0, 93.5, 94.0, 94.5, 95.0, 95.5],
                "ConsistencyScore": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
                "Temperature": [25.0] * 12,
                "Humidity": [50.0] * 12,
                "RainProbability": [0.0] * 12,
                "WindSpeed": [5.0] * 12
            })
    
    def prepare_training_data(self, historical_laps, qualifying_data):
        """
        Prepare training data by merging historical and qualifying data.
        
        Args:
            historical_laps (pd.DataFrame): Historical race data
            qualifying_data (pd.DataFrame): Qualifying session data
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        # Map full names to FastF1 3-letter codes
        driver_mapping = {row["Driver"]: row["DriverCode"] 
                         for _, row in qualifying_data.iterrows()}
        
        # Merge qualifying data with historical race data
        merged_data = qualifying_data.merge(
            historical_laps, 
            left_on="DriverCode", 
            right_on="Driver",
            how="left"
        )
        
        # Add qualifying position as a feature
        merged_data["QualifyingPosition"] = merged_data["QualifyingTime (s)"].rank()
        
        # Calculate gap to leader
        fastest_time = merged_data["QualifyingTime (s)"].min()
        merged_data["QualifyingGapToLeader"] = merged_data["QualifyingTime (s)"] - fastest_time
        
        # Calculate qualifying consistency
        merged_data["QualifyingConsistency"] = 1 / (1 + merged_data["QualifyingTime (s)"].std())
        
        # Add team performance (using qualifying times directly since we don't have team info)
        merged_data["TeamPerformance"] = merged_data["QualifyingTime (s)"]
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in merged_data.columns:
                if feature == "Temperature":
                    merged_data[feature] = 25.0
                elif feature == "Humidity":
                    merged_data[feature] = 50.0
                elif feature == "RainProbability":
                    merged_data[feature] = 0.0
                elif feature == "WindSpeed":
                    merged_data[feature] = 5.0
        
        # Prepare features and target
        X = merged_data[self.feature_names].copy()
        y = merged_data["AvgLapTime (s)"].copy()
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train the prediction model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            
        Returns:
            float: Mean Absolute Error of the model
        """
        # Remove any remaining NaN values
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        return mae
    
    def predict_race(self, qualifying_data):
        """
        Predict race results based on qualifying times and weather data.
        
        Args:
            qualifying_data (pd.DataFrame): Qualifying session data
            
        Returns:
            pd.DataFrame: Predicted race results sorted by predicted race time
        """
        # Convert qualifying_data to DataFrame if it's a dictionary
        if isinstance(qualifying_data, dict):
            qualifying_data = pd.DataFrame(qualifying_data)
        
        # Add qualifying position as a feature
        qualifying_data["QualifyingPosition"] = qualifying_data["QualifyingTime (s)"].rank()
        
        # Calculate gap to leader
        fastest_time = qualifying_data["QualifyingTime (s)"].min()
        qualifying_data["QualifyingGapToLeader"] = qualifying_data["QualifyingTime (s)"] - fastest_time
        
        # Calculate qualifying consistency
        qualifying_data["QualifyingConsistency"] = 1 / (1 + qualifying_data["QualifyingTime (s)"].std())
        
        # Add team performance
        qualifying_data["TeamPerformance"] = qualifying_data["QualifyingTime (s)"]
        
        # Ensure weather data is properly formatted
        weather_columns = ["Temperature", "Humidity", "RainProbability", "WindSpeed"]
        for col in weather_columns:
            if col in qualifying_data.columns:
                # If the column exists but is a single value, broadcast it to all rows
                if len(qualifying_data[col].unique()) == 1:
                    qualifying_data[col] = qualifying_data[col].iloc[0]
            else:
                # Set default values if weather data is missing
                if col == "Temperature":
                    qualifying_data[col] = 25.0
                elif col == "Humidity":
                    qualifying_data[col] = 50.0
                elif col == "RainProbability":
                    qualifying_data[col] = 0.0
                elif col == "WindSpeed":
                    qualifying_data[col] = 5.0
        
        # Prepare features
        X_pred = qualifying_data[self.feature_names].copy()
        
        # Make predictions using the pipeline
        predicted_lap_times = self.pipeline.predict(X_pred)
        
        # Add position-based adjustments
        position_gaps = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7])
        predicted_lap_times = predicted_lap_times + position_gaps[:len(predicted_lap_times)]
        
        # Add small random variations to break ties
        random_noise = np.random.normal(0, 0.02, len(predicted_lap_times))
        predicted_lap_times += random_noise
        
        # Add predictions to qualifying data
        qualifying_data["PredictedRaceTime (s)"] = predicted_lap_times
        
        # Sort by predicted race time
        predictions = qualifying_data.sort_values(by="PredictedRaceTime (s)")
        
        return predictions[["Driver", "PredictedRaceTime (s)"]] 