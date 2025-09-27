#pip install pandas scikit-learn openpyxl
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

df = pd.read_excel('caractNums.xlsx', engine='openpyxl', header=None)
y = df.iloc[:, 0]
x = df.iloc[:, 1:]
print(f'x shape: {x.shape}, y shape: {y.shape}')
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)
# print("Means of original x:", np.round(x.mean().values, 2))
# print("Stds of original x:", np.round(x.std().values, 2))
# print("Means of scaled x:", np.round(np.mean(x_scaled, axis=0), 2))
# print("Stds of scaled x:", np.round(np.std(x_scaled, axis=0), 2))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model_mlp = MLPClassifier(hidden_layer_sizes=(10, 50, 20), max_iter=1000, activation='relu', solver='adam', random_state=42)
model_mlp.fit(x_train, y_train)

y_pred = model_mlp.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}\nClassification Report:\n{classification_report(y_test, y_pred)}')