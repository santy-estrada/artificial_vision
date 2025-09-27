import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os 

# Carga y preparación de datos
df = pd.read_excel(r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\patrones\caractLetras_placas.xlsx', header=None, engine='openpyxl')
y = df.iloc[:, 0]
x = df.iloc[:, 1:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Entrenamiento del Modelo MLP
model_mlp = MLPClassifier(hidden_layer_sizes=(50,100,50), max_iter=20000, activation='relu', solver='adam', random_state=42)
print("Entrenando el modelo...")
model_mlp.fit(x_train, y_train)
print("Entrenamiento completo.")

RUTA_MODELO = 'Proyecto_deteccion_placas/modelos'
os.makedirs(RUTA_MODELO, exist_ok=True) 

# Unir la ruta y el nombre del archivo para crear la ruta completa
nombre_archivo = 'model_letras_placas.joblib'
ruta_completa = os.path.join(RUTA_MODELO, nombre_archivo)

# Guardar el modelo en la ruta completa
joblib.dump(model_mlp, ruta_completa)
print(f"¡Modelo guardado exitosamente en: {ruta_completa}")

# Evaluación del modelo
y_predict = model_mlp.predict(x_test)
print("\n--- Resultados de la Evaluación ---")
print("Accuracy:", accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict, zero_division = 0))