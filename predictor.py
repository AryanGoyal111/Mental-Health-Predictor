import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("mental_health.csv")

# Prepare data
X = data[['Age', 'SleepHours', 'WorkHours', 'MoodScore', 'AnxietyScore']]
y = data['StressLevel'].map({'Low': 0, 'High': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Sample prediction
print("Sample Prediction:", model.predict([[23, 6, 10, 3, 4]]))  # Replace with custom values
