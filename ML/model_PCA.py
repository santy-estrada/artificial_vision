#pip install pandas scikit-learn openpyxl
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.svm import SVC

df = pd.read_excel('caractNums.xlsx', engine='openpyxl', header=None)
y = df.iloc[:, 0]
x = df.iloc[:, 1:]
print(f'x shape: {x.shape}, y shape: {y.shape}')

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=0.95)  # Retain 0.95 of variance
x_pca = pca.fit_transform(x_scaled)


x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.1, random_state=42)

model_svm = SVC(kernel='rbf', C=1.0, random_state=42)
model_svm.fit(x_train, y_train)

y_pred = model_svm.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}\nClassification Report:\n{classification_report(y_test, y_pred)}')

#Show a specific prediction for a sample from y_test
sample_index = 0  # Change this index to test different samples
sample = x_test[sample_index].reshape(1, -1)
predicted_class = model_svm.predict(sample)
actual_class = y_test.iloc[sample_index]
print(f'Sample index: {sample_index}, Predicted class: {predicted_class[0]}, Actual class: {actual_class}')