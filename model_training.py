# 🤖 model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- 1. Synthetic Data Generation ---
# We'll create a realistic dataset to train our model.
print("Generating synthetic data...")
np.random.seed(42)
num_samples = 2000

data = {
    'age': np.random.randint(25, 81, num_samples),
    'sex': np.random.choice([0, 1], num_samples, p=[0.45, 0.55]), # 0: Female, 1: Male
    'cholesterol': np.random.randint(150, 401, num_samples),
    'blood_pressure_systolic': np.random.randint(90, 201, num_samples),
    'blood_pressure_diastolic': np.random.randint(60, 121, num_samples),
    'heart_rate': np.random.randint(50, 121, num_samples),
    'bmi': np.round(np.random.uniform(18.0, 40.0, num_samples), 2),
    'smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
    'family_history': np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
}
df = pd.DataFrame(data)

# Create a target variable 'risk_level' based on a combination of factors
# This logic makes the dataset more realistic and learnable.
risk_score = (
    (df['age'] - 25) / 55 * 0.25 +
    (df['cholesterol'] - 150) / 250 * 0.20 +
    (df['blood_pressure_systolic'] - 90) / 110 * 0.20 +
    df['bmi'] / 40 * 0.15 +
    df['smoking'] * 0.10 +
    df['family_history'] * 0.10
)

# Categorize risk into 3 levels: 0 (Low), 1 (Medium), 2 (High)
df['risk_level'] = pd.cut(risk_score, bins=[-1, 0.4, 0.7, 2], labels=[0, 1, 2]).astype(int)

# --- 2. Data Preprocessing ---
# Introduce some missing values to simulate real-world data
print("Preprocessing data...")
for col in ['cholesterol', 'bmi', 'heart_rate']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# Split features (X) and target (y)
X = df.drop('risk_level', axis=1)
y = df['risk_level']

# Handle missing values using median imputation
# Median is more robust to outliers than mean.
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data using StandardScaler
# This ensures all features contribute equally to the model's performance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Model Training ---
print("Training the model...")
# RandomForestClassifier is a great choice for this type of tabular data.
# It's an ensemble model, robust, and less prone to overfitting.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# --- 4. Model Evaluation ---
print("Evaluating the model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk'])

print(f"\nModel Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# --- 5. Save the Model and Scaler ---
# We save the trained model and scaler to be used by the Streamlit app.
print("Saving model and scaler...")
joblib.dump(model, 'heart_risk_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(imputer, 'imputer.joblib')
df.to_csv('synthetic_heart_data.csv', index=False) # Save data for app display

print("\n✅ Model training complete and artifacts saved successfully!")
print("Run the Streamlit app with: streamlit run streamlit_app.py")