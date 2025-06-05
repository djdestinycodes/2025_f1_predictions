"""
F1 2025 Race Predictions
-----------------------
This program predicts race results for F1 2025 races using historical data, qualifying times,
and weather data. It uses machine learning to predict race outcomes based on various factors
including qualifying performance, historical race data, driver consistency, and weather conditions.
"""

import os
import sys
import pandas as pd

# Add the parent directory to the Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.race_predictor import RacePredictor
from data.race_config import RACES

def display_available_races():
    """
    Display a formatted list of available races for prediction.
    
    Returns:
        None
    """
    print("\n🏎️ Available Races for Prediction:")
    print("-" * 40)
    for race_id, race_info in RACES.items():
        print(f"• {race_id.title()}: {race_info['name']}")
    print("-" * 40)

def get_race_selection():
    """
    Get and validate user's race selection.
    
    Returns:
        str or None: Selected race ID if valid, None if user wants to quit
    """
    while True:
        race_id = input("\nEnter race name (or 'quit' to exit): ").lower()
        if race_id == 'quit':
            return None
        if race_id in RACES:
            return race_id
        print("Invalid race name. Please try again.")

def display_weather_info(race_info):
    """
    Display weather information for the selected race.
    
    Args:
        race_info (dict): Race information dictionary
        
    Returns:
        None
    """
    print("\n🌤️ Weather Conditions:")
    print("-" * 40)
    print(f"Temperature: {race_info['qualifying_data']['Temperature']}°C")
    print(f"Humidity: {race_info['qualifying_data']['Humidity']}%")
    print(f"Rain Probability: {race_info['qualifying_data']['RainProbability'] * 100:.0f}%")
    print(f"Wind Speed: {race_info['qualifying_data']['WindSpeed']} km/h")
    print("-" * 40)

def predict_race(predictor, race_id):
    """
    Make predictions for a specific race.
    
    Args:
        predictor (RacePredictor): Instance of the race predictor
        race_id (str): ID of the race to predict
        
    Returns:
        None
    """
    race_info = RACES[race_id]
    print(f"\n📊 Predicting {race_info['name']}...")
    
    try:
        # Display weather information
        display_weather_info(race_info)
        
        # Load historical data for training
        historical_laps = predictor.load_historical_data(2024, race_info['race_number'])
        
        # Prepare qualifying data with weather information
        qualifying_data = pd.DataFrame({
            "Driver": race_info['qualifying_data']['Driver'],
            "QualifyingTime (s)": race_info['qualifying_data']['QualifyingTime (s)'],
            "DriverCode": race_info['qualifying_data']['DriverCode'],
            "Temperature": race_info['qualifying_data']['Temperature'],
            "Humidity": race_info['qualifying_data']['Humidity'],
            "RainProbability": race_info['qualifying_data']['RainProbability'],
            "WindSpeed": race_info['qualifying_data']['WindSpeed']
        })
        
        # Prepare training data and train model
        X, y = predictor.prepare_training_data(historical_laps, qualifying_data)
        mae = predictor.train_model(X, y)
        
        # Make predictions
        predictions = predictor.predict_race(qualifying_data)
        
        # Display results
        print(f"\n🏆 Predicted Results for {race_info['name']}:")
        print("-" * 50)
        for i, (_, row) in enumerate(predictions.iterrows(), 1):
            print(f"{i}. {row['Driver']}: {row['PredictedRaceTime (s)']:.2f}s")
        print("-" * 50)
        print(f"📈 Model Error (MAE): {mae:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error predicting race: {str(e)}")

def main():
    """
    Main program entry point.
    Handles the main program loop and user interaction.
    """
    print("🏁 Welcome to F1 2025 Race Predictions! 🏁")
    
    # Initialize predictor with cache directory
    predictor = RacePredictor(cache_dir="data/f1_cache")
    
    while True:
        # Display available races
        display_available_races()
        
        # Get user's race selection
        race_id = get_race_selection()
        if race_id is None:
            break
            
        # Make predictions for selected race
        predict_race(predictor, race_id)
        
        # Ask if user wants to predict another race
        if input("\nPredict another race? (y/n): ").lower() != 'y':
            break
    
    print("\nThank you for using F1 2025 Race Predictions! 🏎️")

if __name__ == "__main__":
    main() 