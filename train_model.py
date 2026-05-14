import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("dataset/predictive_maintenance.csv")

# Display first rows
print(data.head())

# Drop unnecessary columns
data.drop(["date", "device"], axis=1, inplace=True)

# Features and target
X = data.drop("failure", axis=1)
y = data["failure"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
pickle.dump(model, open("models/model.pkl", "wb"))

print("\nModel saved successfully!")