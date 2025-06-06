import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Example: synthetic training data (for demo purpose)
# Features: [sleep_hours, work_hours, anxiety_level, mood_score]
X_train = np.array([
    [7, 8, 2, 7],
    [6, 10, 5, 4],
    [8, 7, 1, 8],
    [5, 12, 7, 3]
])
y_train = np.array([0, 1, 0, 1])  # 0: Low stress, 1: High stress

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# New test input to predict
test_input = np.array([[6, 9, 4, 5]])  # e.g., 6 hrs sleep, 9 hrs work, anxiety 4, mood 5

# Make prediction
prediction = model.predict(test_input)

# Map prediction to label
stress_label = 'High Stress' if prediction[0] == 1 else 'Low Stress'

print(f"Test input: {test_input[0]}")
print(f"Predicted stress level: {stress_label}")
