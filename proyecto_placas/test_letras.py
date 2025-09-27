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
    
    # 3. Características de densidad de píxeles por zona
    zone_features = []
    cell_size = 5
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

# Renombramos la función para que sea más genérica
def predecir_caracter(ruta_imagen, modelo, mapeo_clases):

    img_gray = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return None, None

    contour = extract_main_contour(img_gray)
    
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        img_roi_gray = img_gray[y:y+h, x:x+w]
        img_roi_resized = cv2.resize(img_roi_gray, (20, 20), interpolation=cv2.INTER_NEAREST)
        features_vector = extract_features(img_roi_resized, contour)
        features_for_prediction = features_vector.reshape(1, -1)
        
        # El modelo predice la CLASE NUMÉRICA (ej: 0, 1, 2...)
        prediccion_numerica = modelo.predict(features_for_prediction)
        probabilidades = modelo.predict_proba(features_for_prediction)
        
        # Traducimos el resultado numérico al carácter correspondiente
        indice_predicho = int(prediccion_numerica[0])
        
        # Asegurarnos de que el índice no esté fuera del rango del mapeo
        if indice_predicho < len(mapeo_clases):
            caracter_predicho = mapeo_clases[indice_predicho]
        else:
            caracter_predicho = "Error: Clase desconocida"

        # Devolvemos el CARÁCTER y la confianza
        return caracter_predicho, np.max(probabilidades)
        
    else:
        print(f"Advertencia: No se detectó ningún carácter en la imagen '{ruta_imagen}'.")
        return None, None

# ==========================================================================================
#  PASO 3: EJECUCIÓN DEL SCRIPT
# ==========================================================================================
if __name__ == "__main__":
    ### NUEVO ###
    # Define el mapeo de las clases. El orden DEBE ser el mismo usado en el entrenamiento.
    # Si tus carpetas eran 'A', 'B', 'C', etc., este es el orden correcto.
    CLASES_LETRAS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                     'U', 'V', 'W', 'X', 'Y', 'Z']

    # Define la ruta donde está guardado el modelo de LETRAS
    RUTA_MODELO = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\modelos\model_letras_placas.joblib'
    
    # Define la imagen a predecir
    IMAGEN_A_PREDECIR = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\dataset_caracteres\Z\3.png' 

    if os.path.exists(RUTA_MODELO):
        modelo_cargado = joblib.load(RUTA_MODELO)
        print(f"Modelo '{os.path.basename(RUTA_MODELO)}' cargado exitosamente.")
        
        ### MODIFICADO ###
        # Llamamos a la función con el mapeo de clases
        resultado, confianza = predecir_caracter(IMAGEN_A_PREDECIR, modelo_cargado, CLASES_LETRAS)
        
        if resultado is not None:
            print("\n==============================================")
            ### MODIFICADO ###
            # Actualizamos el mensaje de salida
            print(f"  El caracter predicho es: {resultado}")
            print(f"  Confianza de la predicción: {confianza:.2%}")
            print("==============================================")
            
    else:
        print(f"Error: No se encontró el archivo del modelo en '{RUTA_MODELO}'.")
        print("Asegúrate de haber ejecutado primero el script de entrenamiento y guardado.")