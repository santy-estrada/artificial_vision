import cv2
import numpy as np
from glob import glob
import xlsxwriter
import os


BASE_PATH = "Proyecto_deteccion_placas"
DATASET_PATH = os.path.join(BASE_PATH, "dataset_caracteres")
OUTPUT_DIR = os.path.join(BASE_PATH, "patrones")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "caractLetras_placas.xlsx")

# Asegurarse de que el directorio de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    for y in range(0, img_roi_bin.shape[0], cell_size):
        for x in range(0, img_roi_bin.shape[1], cell_size):
            roi_zone = img_roi_bin[y:y+cell_size, x:x+cell_size]
            density = cv2.countNonZero(roi_zone) / (cell_size * cell_size)
            zone_features.append(density)
            
    # Concatenar todas las características en un solo vector
    all_features = np.concatenate(([area, perimeter, circularity], hu_moments, zone_features))
    
    return all_features.astype(np.float32)

def process_dataset():
    
    vector_folders_nums = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    all_data_for_excel = []

    print("Iniciando extracción de características...")

    for n, folder_num in enumerate(vector_folders_nums):
        # Usamos os.path.join para construir la ruta de forma segura
        search_path = os.path.join(DATASET_PATH, folder_num, "*.png")
        
        for img_path in glob(search_path):
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"Advertencia: No se pudo leer la imagen {img_path}")
                continue

            contour = extract_main_contour(img_gray)
            
            if contour is not None:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extraer la región de interés (ROI) que contiene el número
                img_roi = img_gray[y:y+h, x:x+w]
                
                # **PASO CLAVE**: Redimensionar el ROI a un tamaño estándar (20x20)
                # Esto garantiza que las características de zona siempre se calculen sobre una misma base.
                img_roi_resized = cv2.resize(img_roi, (20, 20), interpolation=cv2.INTER_NEAREST)
                
                # Extraer el vector de características
                features_vector = extract_features(img_roi_resized, contour)
                
                # Añadir la etiqueta (el número que es) al inicio del vector de características
                labeled_features = np.insert(features_vector, 0, n)
                all_data_for_excel.append(labeled_features)

                # Visualización (opcional)
                cv2.imshow("ROI Redimensionada", img_roi_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
    
    cv2.destroyAllWindows()

    # --- ESCRITURA EN EXCEL (se hace una sola vez al final para mayor eficiencia) ---
    if not all_data_for_excel:
        print("No se extrajo ninguna característica. El archivo Excel no se creará.")
        return

    print(f"Extracción finalizada. Escribiendo {len(all_data_for_excel)} registros en {OUTPUT_FILE}...")
    
    workbook = xlsxwriter.Workbook(OUTPUT_FILE)
    worksheet = workbook.add_worksheet('caracteristicas')
    
    for row_idx, data_row in enumerate(all_data_for_excel):
        for col_idx, cell_data in enumerate(data_row):
            worksheet.write(row_idx, col_idx, cell_data)
            
    workbook.close()
    print("¡Proceso completado con éxito!")

# --- Ejecutar el proceso ---
if __name__ == "__main__":
    process_dataset()