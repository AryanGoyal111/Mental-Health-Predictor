# 🧠 Mental Health Predictor using Machine Learning

This project predicts stress levels (Low/High) based on survey data about age, sleep, work hours, mood, and anxiety scores using a Random Forest Classifier.

## 🚀 Technologies Used

- Python
- Pandas
- Scikit-learn
- RandomForestClassifier

## 📊 Dataset Fields

- `Age`
- `SleepHours`
- `WorkHours`
- `MoodScore` (1–5)
- `AnxietyScore` (1–5)
- `StressLevel` (Target: Low/High)

## 🔍 How it works

1. Load data from CSV
2. Split into training and testing
3. Train Random Forest model
4. Evaluate and predict

## 🧪 Sample Prediction

```python
print(model.predict([[23, 6, 10, 3, 4]]))
