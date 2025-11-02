import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import joblib
import math

# =====================
# 1. LOAD DATA
# =====================
df = pd.read_csv('cabdata.csv')
print("✅ Data loaded successfully!")

# =====================
# 2. DATA CLEANING
# =====================
# Keep only relevant columns for fare prediction
df = df[[
    'month', 'day_of_week', 'passenger_count',
    'model', 'Time_Category',
    'Delhi_latitude', 'City_longitude',
    'dropoff_latitude', 'dropoff_longitude', 'fare_amount'
]]

# Drop rows with missing or invalid fare values
df = df.dropna(subset=['fare_amount'])
df = df[df['fare_amount'] > 0]
df = df[df['passenger_count'] > 0]

# Rename columns for consistency
df = df.rename(columns={
    'Delhi_latitude': 'pickup_latitude',
    'City_longitude': 'pickup_longitude',
    'Time_Category': 'time_category'
})

print("✅ Data cleaned successfully!")

# =====================
# 3. ENCODE CATEGORICAL COLUMNS
# =====================
model_encoder = LabelEncoder()
day_of_week_encoder = LabelEncoder()
time_category_encoder = LabelEncoder()

df['model'] = model_encoder.fit_transform(df['model'])
df['day_of_week'] = day_of_week_encoder.fit_transform(df['day_of_week'])
df['time_category'] = time_category_encoder.fit_transform(df['time_category'])

print("✅ Categorical columns encoded successfully!")

# =====================
# 4. FEATURE ENGINEERING – ADD DISTANCE
# =====================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

print("📏 Added calculated distance feature successfully!")

# =====================
# 5. FEATURE SELECTION
# =====================
features = [
    'month', 'day_of_week', 'passenger_count',
    'model', 'time_category',
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude', 'distance_km'
]
target = 'fare_amount'

X = df[features]
y = df[target]

# =====================
# 6. TRAIN-TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split into train ({len(X_train)}) and test ({len(X_test)}) sets!")

# =====================
# 7. MODEL TRAINING
# =====================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training completed!")

# =====================
# 8. MODEL EVALUATION
# =====================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:")
print(f"   Mean Absolute Error (MAE): {mae:.2f}")
print(f"   R² Score: {r2:.4f}")

# =====================
# 9. SAVE MODEL AND ENCODERS
# =====================
joblib.dump(model, 'best_model.pkl')
print("💾 Model saved as best_model.pkl")

with open('model_encoder.pkl', 'wb') as f:
    pickle.dump(model_encoder, f)
with open('day_of_week_encoder.pkl', 'wb') as f:
    pickle.dump(day_of_week_encoder, f)
with open('time_category_encoder.pkl', 'wb') as f:
    pickle.dump(time_category_encoder, f)

print("💾 LabelEncoders saved successfully!")

# =====================
# 10. COMPLETION MESSAGE
# =====================
print("🎉 Training pipeline completed successfully!")
