"""
Race Configuration Module
------------------------
This module contains the configuration data for F1 2025 races, including race names,
race numbers, and qualifying data for each race.
"""

# Race configurations for F1 2025 season
RACES = {
    "australia": {
        "name": "Australian Grand Prix",
        "race_number": 3,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                75.096, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "NOR", "PIA", "VER", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 22.0,  # Celsius
            "Humidity": 65.0,     # Percentage
            "RainProbability": 0.1,  # 0-1 scale
            "WindSpeed": 12.0     # km/h
        }
    },
    "china": {
        "name": "Chinese Grand Prix",
        "race_number": 5,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                75.096, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "NOR", "PIA", "VER", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 18.0,  # Celsius
            "Humidity": 70.0,     # Percentage
            "RainProbability": 0.3,  # 0-1 scale
            "WindSpeed": 8.0      # km/h
        }
    },
    "japan": {
        "name": "Japanese Grand Prix",
        "race_number": 4,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                74.996, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 20.0,  # Celsius
            "Humidity": 75.0,     # Percentage
            "RainProbability": 0.4,  # 0-1 scale
            "WindSpeed": 15.0     # km/h
        }
    },
    "miami": {
        "name": "Miami Grand Prix",
        "race_number": 6,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Carlos Sainz", "Alexander Albon", "Charles Leclerc", "Esteban Ocon",
                "Yuki Tsunoda", "Lewis Hamilton", "Lance Stroll", "Pierre Gasly",
                "Fernando Alonso", "Nico Hulkenberg"
            ],
            "QualifyingTime (s)": [
                86.204, 86.269, 86.375, 86.385, 86.569, 86.682, 86.754,
                86.824, 86.943, 87.006, 87.830, 87.710, 87.604, 87.473
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
                "TSU", "HAM", "STR", "GAS", "ALO", "HUL"
            ],
            "Temperature": 28.0,  # Celsius
            "Humidity": 80.0,     # Percentage
            "RainProbability": 0.2,  # 0-1 scale
            "WindSpeed": 10.0     # km/h
        }
    },
    "emilia": {
        "name": "Emilia Romagna Grand Prix",
        "race_number": 7,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                74.996, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 23.0,  # Celsius
            "Humidity": 60.0,     # Percentage
            "RainProbability": 0.3,  # 0-1 scale
            "WindSpeed": 7.0      # km/h
        }
    },
    "monaco": {
        "name": "Monaco Grand Prix",
        "race_number": 8,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Carlos Sainz", "Alexander Albon", "Charles Leclerc", "Esteban Ocon",
                "Yuki Tsunoda", "Lewis Hamilton", "Lance Stroll", "Pierre Gasly",
                "Fernando Alonso", "Nico Hulkenberg"
            ],
            "QualifyingTime (s)": [
                86.204, 86.269, 86.375, 86.385, 86.569, 86.682, 86.754,
                86.824, 86.943, 87.006, 87.830, 87.710, 87.604, 87.473
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
                "TSU", "HAM", "STR", "GAS", "ALO", "HUL"
            ],
            "Temperature": 24.0,  # Celsius
            "Humidity": 65.0,     # Percentage
            "RainProbability": 0.2,  # 0-1 scale
            "WindSpeed": 5.0      # km/h
        }
    },
    "canada": {
        "name": "Canadian Grand Prix",
        "race_number": 9,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                74.996, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 19.0,  # Celsius
            "Humidity": 55.0,     # Percentage
            "RainProbability": 0.4,  # 0-1 scale
            "WindSpeed": 15.0     # km/h
        }
    },
    "spain": {
        "name": "Spanish Grand Prix",
        "race_number": 10,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                74.996, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 26.0,  # Celsius
            "Humidity": 45.0,     # Percentage
            "RainProbability": 0.1,  # 0-1 scale
            "WindSpeed": 12.0     # km/h
        }
    },
    "austria": {
        "name": "Austrian Grand Prix",
        "race_number": 11,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                74.996, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 22.0,  # Celsius
            "Humidity": 60.0,     # Percentage
            "RainProbability": 0.3,  # 0-1 scale
            "WindSpeed": 10.0     # km/h
        }
    },
    "silverstone": {
        "name": "British Grand Prix",
        "race_number": 12,  # Race number in 2024 season
        "qualifying_data": {
            "Driver": [
                "Max Verstappen", "Lando Norris", "Oscar Piastri", "George Russell",
                "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
                "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"
            ],
            "QualifyingTime (s)": [
                74.996, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755,
                75.973, 75.980, 76.062, 76.4, 76.5
            ],
            "DriverCode": [
                "VER", "NOR", "PIA", "RUS", "TSU", "ALB", "LEC", "HAM",
                "GAS", "SAI", "ALO", "STR"
            ],
            "Temperature": 18.0,  # Celsius
            "Humidity": 70.0,     # Percentage
            "RainProbability": 0.5,  # 0-1 scale
            "WindSpeed": 20.0     # km/h
        }
    }
} 