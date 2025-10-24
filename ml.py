import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv('analysis_data_fully_processed.csv')

df_ml = df[df['caption'].str.len() > 10].copy()

X = df_ml['caption']
y = df_ml['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("\nReport")
print(classification_report(y_test, y_pred))

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

feature_importance = pd.Series(coefficients, index=feature_names)

top_political_features = feature_importance.sort_values(ascending=False).head(10)
top_entertainment_features = feature_importance.sort_values(ascending=True).head(10)

with open('results_C_ml_report.txt', 'w') as f:
    f.write("--- report---\n")
    f.write(classification_report(y_test, y_pred))

    f.write("\n\n--- Top 10 Keywords that predict the Political Category ---\n")
    f.write(top_political_features.to_string() + "\n")

    f.write("\n--- Top 10 Keywords that predict the Entertainment Category ---\n")
    f.write(top_entertainment_features.to_string() + "\n")

print("saved")
