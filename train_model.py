import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv('bank_data.csv', sep=';')

# =========================
# DATA PREPROCESSING
# =========================

# Convert categorical columns into numeric
label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# =========================
# FEATURES & TARGET
# =========================
X = df.drop('y', axis=1)
y = df['y']

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# MODEL EVALUATION
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, 'model.pkl')

print("Model Saved Successfully")