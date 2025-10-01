#pip install pandas scikit-learn openpyxl
from sklearn.model_selection import train_test_split
import pandas as pd
# Import additional metrics for detailed evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import GridSearchCV


df = pd.read_excel('caractTeeth.xlsx', engine='openpyxl', header=None)
y = df.iloc[:, 0]
x = df.iloc[:, 1:]
print(f'x shape: {x.shape}, y shape: {y.shape}')

# Create a pipeline with scaler, PCA, and MLP
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Retain 95% of variance
    ('mlp', MLPClassifier(max_iter=10000, random_state=42))
])

# Define parameter grid for GridSearchCV (MLP parameters only)
param_grid = {
    'mlp__hidden_layer_sizes': [(10,), (50,), (100,), (10, 50), (50, 20), (10, 50, 20), (100, 50), (100, 100)],
    'mlp__activation': ['relu', 'tanh', 'logistic'],
    'mlp__solver': ['adam', 'lbfgs', 'sgd'],
    'mlp__learning_rate_init': [0.001, 0.01, 0.1],
    'mlp__alpha': [0.0001, 0.001, 0.01, 0.1]  # L2 penalty (regularization)
}

# Split the data before fitting
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=42)

# Perform GridSearchCV
print("Starting GridSearchCV...")
grid_search = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    cv=3,  # 3-fold cross-validation
    scoring='f1_weighted',  # Use f1_weighted for classification (matches caso_clasificacion.ipynb)
    n_jobs=-1,  # Use all available cores
    verbose=1  # Show progress
)

# Fit the grid search
grid_search.fit(x_train, y_train)

# Get the best model
best_pipeline = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score (F1-weighted): {grid_search.best_score_:.4f}")

# Show PCA information after fitting
pca_component = best_pipeline.named_steps['pca']
print(f'Number of components after PCA: {pca_component.n_components_}')
print(f'Explained variance ratio: {pca_component.explained_variance_ratio_}')
print(f'Total explained variance: {sum(pca_component.explained_variance_ratio_):.4f}')

# Make predictions with the best model
y_pred_train = best_pipeline.predict(x_train)
y_pred_test = best_pipeline.predict(x_test)

# Performance metrics for training set
print('\n' + '='*50)
print('MÉTRICAS DE DESEMPEÑO EN ENTRENAMIENTO')
print('='*50)
print(classification_report(y_train, y_pred_train))
print(f'Accuracy en entrenamiento: {accuracy_score(y_train, y_pred_train):.4f}')

# Confusion matrix for training (printed in terminal)
print('\nMatriz de Confusión - Entrenamiento:')
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)

# Commented out seaborn plot to avoid manual window closing
# plt.figure(figsize=(8, 6))
# sb.heatmap(confusion_matrix(y_train, y_pred_train), annot=True, fmt='d', cmap='Blues')
# plt.title('Matriz de Confusión - Entrenamiento')
# plt.ylabel('Valores Reales')
# plt.xlabel('Predicciones')
# plt.show()

# Performance metrics for test set
print('\n' + '='*50)
print('MÉTRICAS DE DESEMPEÑO EN PRUEBA')
print('='*50)
print(classification_report(y_test, y_pred_test))
print(f'Accuracy en prueba: {accuracy_score(y_test, y_pred_test):.4f}')

# Confusion matrix for test (printed in terminal)
print('\nMatriz de Confusión - Prueba:')
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

# Commented out seaborn plot to avoid manual window closing
# plt.figure(figsize=(8, 6))
# sb.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues')
# plt.title('Matriz de Confusión - Prueba')
# plt.ylabel('Valores Reales')
# plt.xlabel('Predicciones')
# plt.show()

# Display validation results (top 10 best combinations)
validation_results = pd.DataFrame(grid_search.cv_results_)
validation_summary = validation_results[['params', 'mean_test_score']].rename(columns={'mean_test_score': 'f1_weighted'}).sort_values(by='f1_weighted', ascending=False)
print("\nTop 10 parameter combinations:")
print(validation_summary.head(10))

# Save the best pipeline
joblib.dump(best_pipeline, 'teeth_classification_mlp_best.pkl')
print("\nBest pipeline saved as 'teeth_classification_mlp_best.pkl'")