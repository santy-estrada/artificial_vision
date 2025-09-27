import cv2
import numpy as np
import joblib
import os

def extract_main_contour(img_binary): 
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    
    # Filtra contornos muy pequeños que podrían ser ruido
    if cv2.contourArea(main_contour) > 50:
        return main_contour
    return None

def extract_features(img_roi_bin, contour):
    
    # 1. Características de forma del contorno
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    # Evitar división por cero si el perímetro es 0
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    
    # 2. Momentos de Hu (invariantes a escala, rotación y traslación)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # 3. Características de densidad de píxeles por zona (dividiendo la imagen 20x20 en 16 zonas de 5x5)
    zone_features = []
    cell_size = 5
    # cv2.countNonZero tratará cualquier píxel > 0 como no-cero, funcionando para grises.
    for y in range(0, img_roi_bin.shape[0], cell_size):
        for x in range(0, img_roi_bin.shape[1], cell_size):
            roi_zone = img_roi_bin[y:y+cell_size, x:x+cell_size]
            density = cv2.countNonZero(roi_zone) / (cell_size * cell_size)
            zone_features.append(density)
            
    # Concatenar todas las características en un solo vector
    all_features = np.concatenate(([area, perimeter, circularity], hu_moments, zone_features))
    
    return all_features.astype(np.float32)

# ==========================================================================================
#  PASO 2: FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ==========================================================================================

def predecir_numero(ruta_imagen, modelo):

    # 1. Cargar la imagen en escala de grises (como en el script de entrenamiento)
    img_gray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return None

    # 2. Encontrar contorno directamente en la imagen en grises (como en el script de entrenamiento)
    contour = extract_main_contour(img_gray)
    
    # 3. Si se encuentra un contorno válido
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Recortar la región de interés (ROI) de la imagen en grises (como en el script de entrenamiento)
        img_roi_gray = img_gray[y:y+h, x:x+w]
        
        # Redimensionar a 20x20
        img_roi_resized = cv2.resize(img_roi_gray, (20, 20), interpolation=cv2.INTER_NEAREST)
        
        # 4. Extraer las características usando el contorno original y el ROI redimensionado
        features_vector = extract_features(img_roi_resized, contour)
        
        # 5. Formatear para la predicción (sklearn espera un array 2D)
        features_for_prediction = features_vector.reshape(1, -1)
        
        # 6. Realizar la predicción y obtener probabilidades
        prediccion = modelo.predict(features_for_prediction)
        probabilidades = modelo.predict_proba(features_vector.reshape(1, -1))
        
        # Devolver el resultado y la confianza
        return prediccion[0], np.max(probabilidades)
        
    else:
        print(f"Advertencia: No se detectó ningún número en la imagen '{ruta_imagen}'.")
        return None, None

# ==========================================================================================
#  PASO 3: EJECUCIÓN DEL SCRIPT
# ==========================================================================================
if __name__ == "__main__":
    # Define la ruta donde está guardado el modelo
    RUTA_MODELO = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\modelos\model_num_placas.joblib'
    
    IMAGEN_A_PREDECIR = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\dataset_caracteres\7\43.png' 

    # Cargar el modelo
    if os.path.exists(RUTA_MODELO):
        modelo_cargado = joblib.load(RUTA_MODELO)
        print(f"Modelo '{RUTA_MODELO}' cargado exitosamente.")
        
        # Realizar la predicción
        resultado, confianza = predecir_numero(IMAGEN_A_PREDECIR, modelo_cargado)
        
        # Mostrar el resultado final
        if resultado is not None:
            print("\n==============================================")
            print(f"  El número predicho es: {int(resultado)}")
            print(f"  Confianza de la predicción: {confianza:.2%}")
            print("==============================================")
            
    else:
        print(f"Error: No se encontró el archivo del modelo en '{RUTA_MODELO}'.")
        print("Asegúrate de haber ejecutado primero el script de entrenamiento y guardado.")