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
print(f'PCA reduced x shape: {x_pca.shape}')
expl = pca.explained_variance_ratio_
print(f'Explained variance ratio by each component: {expl}')
n_components = 34

for i in range(1, n_components + 1):
    print(f'Variance explained by first {i} components: {np.sum(expl[:i])}')
    #Grtaph the variance explained by each component
    
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
cumulative_variance = [np.sum(expl[:i]) for i in range(1, n_components + 1)]
plt.plot(range(1, n_components + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.1, random_state=42)

model_svm = SVC(kernel='rbf', C=1.0, random_state=42)
model_svm.fit(x_train, y_train)

y_pred = model_svm.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}\nClassification Report:\n{classification_report(y_test, y_pred)}')