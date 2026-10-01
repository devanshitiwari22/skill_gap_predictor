import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

np.random.seed(42)
n = 1000

cosine_sim = np.random.uniform(0.2, 0.95, n)
coverage = np.clip(cosine_sim + np.random.normal(0, 0.08, n), 0.1, 1.0)
missing_count = np.clip(np.round((1 - coverage) * 8), 0, 8)
years_left = np.random.choice([0, 1, 2, 3], size=n, p=[0.3, 0.4, 0.2, 0.1])
future_weight = np.random.uniform(0.1, 0.9, n)

score = (0.35 * coverage) + (0.30 * cosine_sim) + (0.20 * future_weight) - (0.05 * missing_count) + (0.05 * years_left)

labels = []
for s in score:
    if s >= 0.50:
        labels.append(2)
    elif s >= 0.32:
        labels.append(1)
    else:
        labels.append(0)

X = pd.DataFrame({
    'cosine_sim': cosine_sim,
    'coverage': coverage,
    'missing_count': missing_count,
    'years_left': years_left,
    'future_weight': future_weight
})
y = pd.Series(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("5-Fold Cross Validation Accuracy:", scores.mean() * 100)
print("Test Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "skill_evaluator_rf.pkl")
print("Model saved successfully as skill_evaluator_rf.pkl")