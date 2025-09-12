import threading
from unicodedata import name
import cv2
from logger import Logger
import time


class RunCamera():
    def __init__(self, src=0, name="Camera_1"):
        self.loggerReport = Logger("logCamera")

        try:
            self.src = src
            self.name = name
            self.loggerReport.logger.info("Init constructor RunCamera")
            self.ret = None
            self.frame = None
            self.processed_frame = None  # Store processed frame for detection display
            self.stopped = False
            self.cont1 = 0
            self.cont2 = 0
            self.flag = False
            self.testigo = True
            self.detection_occurred = False  # Flag to indicate new detection
        except Exception as e:
            self.loggerReport.logger.error("Error initializing RunCamera: " + str(e))

    def start_camera(self):
        try:
            self.stream = cv2.VideoCapture(self.src)
            time.sleep(1)
            self.ret, self.frame = self.stream.read()
            if (self.stream.isOpened()):
                self.loggerReport.logger.info("Creating Thread in start_camera")
                self.my_thread = threading.Thread(target=self.get, name=self.name, daemon=True)
                self.my_thread.start()
                self.stopped = False
            else:
                self.loggerReport.logger.warning("Start_camera not initialized")
        except Exception as e:
            self.loggerReport.logger.error("Error starting camera: " + str(e))

    def stop(self):
        self.stopped = True
        if self.stream is not None:
            self.stream.release()
        self.loggerReport.logger.info("Camera stopped")

    def is_playing(self):
        """Check if the video/camera is still active"""
        return not self.stopped and self.stream is not None and self.stream.isOpened()

    def get(self):
        while not self.stopped:
            try:
                self.ret, self.frame = self.stream.read()
                if self.ret:
                    time.sleep(0.02)
                    self.process_image()
                if not self.ret:
                    # Video has ended
                    print(f"Video '{self.src}' has ended.")
                    self.loggerReport.logger.info(f"Video '{self.src}' has ended.")
                    break
            except Exception as e:
                self.loggerReport.logger.error("Error in get Camera: " + str(e))
                    
    def process_image(self):
        # Create a copy of the frame for processing
        processed_frame = self.frame.copy()

        hsv_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        img_binary = cv2.inRange(hsv_frame, (80, 0, 40), (115, 255, 255))
        img_binary= cv2.medianBlur(img_binary, 5)  #Params: (image, ksize --> must be odd)

        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Params: (image, mode, method)
        contour_areas = [cv2.contourArea(item) for item in contours]
        max_area = max(contour_areas) if contour_areas else 0
        max_area_contour = contours[contour_areas.index(max_area)] if contour_areas else None
        x,y,w,h = cv2.boundingRect(max_area_contour) if max_area_contour is not None else (0,0,0,0)
        
        # Draw all significant contours on the processed frame
        for contour in contours:
            area = cv2.contourArea(contour)
            if x < 300 and self.testigo:
                if area == max_area and area > 5000:
                    cv2.rectangle(img_binary, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    self.testigo = False
                    self.flag = True

            elif x > 400:
                self.testigo = True

            if self.flag:
                numCont = len(contours)
                self.flag = False
                large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
                if len(large_contours) == 2:
                    self.cont1 += 1
                    # Store the processed frame when detection occurs
                    self.processed_frame = img_binary.copy()
                    self.detection_occurred = True
                    print(f"Pieza_1 detected! Count: {self.cont1}")
                elif len(large_contours) == 3:
                    self.cont2 += 1
                    # Store the processed frame when detection occurs
                    self.processed_frame = img_binary.copy()
                    self.detection_occurred = True
                    print(f"Pieza_2 detected! Count: {self.cont2}")
