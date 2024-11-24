

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(size=10000):
    # Generate random typing speeds and mouse movements
    typing_speed = np.random.uniform(30, 120, size=size)  # Words per minute
    mouse_movement = np.random.uniform(50, 400, size=size)  # Arbitrary units
    locations = np.random.choice(["New York", "California", "Texas", "Florida", "Washington"], size=size)

    # Create a DataFrame
    df = pd.DataFrame({
        "typing_speed": typing_speed,
        "mouse_movement": mouse_movement,
        "location": locations
    })

    # Encode location as a simple integer
    df["location_encoded"] = df["location"].apply(lambda x: hash(x) % 100)

    # Save the synthetic dataset for debugging if needed
    df.to_csv("data/synthetic_user_data.csv", index=False)

    return df

def preprocess_data():
    # Generate synthetic data
    df = generate_synthetic_data()

    # Extract and encode features
    typing_speed = df["typing_speed"].values.reshape(-1, 1)
    mouse_movement = df["mouse_movement"].values.reshape(-1, 1)
    location_encoded = df["location_encoded"].values.reshape(-1, 1)

    # Concatenate all features
    features = np.hstack((typing_speed, mouse_movement, location_encoded))

    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Save scaler for inference
    np.save("data/scaler_params.npy", scaler.mean_)

    return scaled_features
