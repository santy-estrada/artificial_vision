import threading
import time
import cv2
import numpy as np
from logger import Logger
from ultralytics import YOLO
import joblib
import os

# --- La función extract_features se mantiene igual ---
def extract_features(img_roi_bin, contour):
    try:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        zone_features = []
        cell_size = 5
        for y in range(0, img_roi_bin.shape[0], cell_size):
            for x in range(0, img_roi_bin.shape[1], cell_size):
                roi_zone = img_roi_bin[y:y+cell_size, x:x+cell_size]
                density = cv2.countNonZero(roi_zone) / (cell_size * cell_size)
                zone_features.append(density)
        all_features = np.concatenate(([area, perimeter, circularity], hu_moments, zone_features))
        return all_features.astype(np.float32)
    except Exception as e:
        print(f"Error extrayendo características: {e}")
        return None

class RunCamera():
    def __init__(self, src=0, name="Camera_1"):
        try:
            self.name = name
            self.src = src
            self.stopped = False
            self.loggerReport = Logger("logCamera")
            
            self.video_entrada = None
            self.placa_detectada = None
            self.placa_detectada_var = "---"
            self.total_carros_var = 0

            # ================== NUEVO: ATRIBUTOS DE ESTADO Y COOLDOWN ==================
            self.ultima_placa_leida = ""
            self.tiempo_ultima_lectura = 0
            self.COOLDOWN_SEGUNDOS = 6.0 # Esperar 5 segundos antes de leer otra placa

            self.loggerReport.logger.info("Init constructor RunCamera con clasificador")
            
            self.model = YOLO(r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\bestPlateCar.pt')

            ruta_modelo_letras = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\modelos\model_letras_placas.joblib'
            ruta_modelo_numeros = r'C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\modelos\model_num_placas.joblib'
            
            if os.path.exists(ruta_modelo_letras):
                self.modelo_letras = joblib.load(ruta_modelo_letras)
                self.modelo_numeros = joblib.load(ruta_modelo_numeros)
                self.loggerReport.logger.info("Modelos de reconocimiento cargados exitosamente.")
            else:
                self.modelo_letras = None
                self.modelo_numeros = None

            self.CLASES_LETRAS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                                  'U', 'V', 'W', 'X', 'Y', 'Z']
            # Para los números, el mapeo es el número mismo
            self.CLASES_NUMEROS = [str(i) for i in range(10)]

        except Exception as e:
            self.loggerReport.logger.error("Error en constructor RunCamera: " + str(e))

    def start(self):
        # ... (el método start se mantiene igual) ...
        try:
            self.stream = cv2.VideoCapture(self.src)
            if not self.stream.isOpened():
                self.loggerReport.logger.error(f"Error: No se puede abrir la cámara/video {self.name}")
                return
            
            self.my_thread = threading.Thread(target=self.get, name=self.name, daemon=True)
            self.my_thread.start()
            self.loggerReport.logger.info(f"Cámara {self.name} iniciada correctamente")
        except Exception as e:
            self.loggerReport.logger.error(f"Error al iniciar la cámara: {e}")

    def get(self):
            TARGET_SIZE = (20, 20)

            while self.stream.isOpened() and not self.stopped:
                ret, frame = self.stream.read()
                if not ret:
                    break

                tiempo_actual = time.time()
                placa_a_mostrar = None
                placa_fue_validada = False

                # Comprobar si estamos en período de cooldown
                if tiempo_actual - self.tiempo_ultima_lectura < self.COOLDOWN_SEGUNDOS:
                    self.video_entrada = frame
                    time.sleep(0.015)
                    continue # Saltamos al siguiente fotograma

                # Si salimos del cooldown, reseteamos para buscar una nueva
                self.placa_detectada_var = "---"
                self.placa_detectada = None

                results = self.model.predict(source=frame, conf=0.2, verbose=False)

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        imgRoi = frame[y1:y2, x1:x2]

                        if imgRoi.size > 0:
                            gray_roi = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
                            _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            # Buscar solo contornos externos primero 
                            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                            roi_with_contours = imgRoi.copy()
                            
                            caracteres_detectados = []

                            # Esta lista guardará los rectángulos de los caracteres ya encontrados.
                            cajas_de_caracteres_encontrados = []

                            # Ordenar contornos por tamaño para procesar los más grandes primero ---
                            contours = sorted(contours, key=cv2.contourArea, reverse=True)

                            for cnt in contours:
                                x, y, w, h = cv2.boundingRect(cnt)
                                aspect_ratio = w / h if h > 0 else 0
                                area = cv2.contourArea(cnt)

                                # Comprobamos si el centro de este contorno ya está dentro de una caja detectada.
                                centro_x, centro_y = x + w // 2, y + h // 2
                                ya_detectado = False
                                for caja in cajas_de_caracteres_encontrados:
                                    c_x, c_y, c_w, c_h = caja
                                    if c_x < centro_x < c_x + c_w and c_y < centro_y < c_y + c_h:
                                        ya_detectado = True
                                        break
                                
                                # Si ya fue detectado, lo ignoramos y pasamos al siguiente contorno.
                                if ya_detectado:
                                    continue
                                # ==============================================================================

                                if (0.2 < aspect_ratio < 1.2) and (h < 0.7 * imgRoi.shape[0]) and  (h > 0.35 * imgRoi.shape[0]) and (area > 150) and (area < 800):
                                    # Si es un carácter válido, registramos su caja para no detectar otros dentro.
                                    cajas_de_caracteres_encontrados.append((x, y, w, h))
                                    # ==============================================================================

                                    char_image = mask[y:y+h, x:x+w]
                                    char_resized = cv2.resize(char_image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                                    features = extract_features(char_resized, cnt)

                                    if features is not None:
                                        posicion_relativa = (x + w/2) / imgRoi.shape[1]
                                        
                                        if posicion_relativa < 0.5: # Primeros 3 son letras
                                            modelo, mapeo = self.modelo_letras, self.CLASES_LETRAS
                                        else: # Últimos 3 son números
                                            modelo, mapeo = self.modelo_numeros, self.CLASES_NUMEROS
                                        
                                        pred_num = modelo.predict(features.reshape(1, -1))
                                        confianza = np.max(modelo.predict_proba(features.reshape(1, -1)))

                                        if confianza > 0.5:
                                            caracter = mapeo[int(pred_num[0])]
                                            caracteres_detectados.append({'char': caracter, 'x': x})
                                            # Dibujar en la copia para visualización
                                            cv2.drawContours(roi_with_contours, [cnt], -1, (0, 0, 255), 1) # Dibujar en rojo
                                        
                                            cv2.rectangle(roi_with_contours, (x, y), (x + w, y + h), (255, 0, 0), 1)
                                            cv2.putText(roi_with_contours, caracter, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

                            # Después de analizar todos los contornos, ordenamos y validamos
                            if len(caracteres_detectados) == 6:
                                caracteres_ordenados = sorted(caracteres_detectados, key=lambda item: item['x'])
                                placa_texto = "".join([item['char'] for item in caracteres_ordenados])
                                
                                # Actualizamos el estado
                                self.placa_detectada_var = placa_texto
                                self.tiempo_ultima_lectura = time.time() # Iniciamos el cooldown
                                placa_a_mostrar = roi_with_contours
                                placa_fue_validada = True
                                self.total_carros_var += 1
                                self.loggerReport.logger.info(f"Placa leída exitosamente: {placa_texto}")
                                break 
                    if placa_fue_validada:
                        break

                self.video_entrada = frame
                self.placa_detectada = placa_a_mostrar
                
                time.sleep(0.005)

    def stop(self):
        self.stopped = True
        time.sleep(0.5)
        if hasattr(self, 'stream') and self.stream.isOpened():
            self.stream.release()
        self.loggerReport.logger.info(f"Cámara {self.name} detenida.")