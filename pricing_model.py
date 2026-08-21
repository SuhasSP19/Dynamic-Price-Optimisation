# pricing_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def train_dynamic_pricing_model(data):
    features = ['price', 'remaining_stock', 'quantity_sold', 'num_customers_visited']
    target = 'optimal_price'
    
    if target not in data.columns:
        data[target] = data['price'] * (1 + np.random.uniform(-0.2, 0.2, len(data)))

    X = data[features]
    y = data[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    with open("models/dynamic_pricing_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler

def predict_optimal_price(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)