import threading
import cv2
import time
from logger import Logger
import classifier
import numpy as np


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
            self.cont_unperforated = 0  # 1 large contour in one color
            self.cont_wrong = 0         # Mixed colors or wrong configuration
            self.cont_proper = 0        # 2 large contours in only one color
            self.flag = False
            self.testigo = True
            self.detection_occurred = False  # Flag to indicate new detection
            self.current_classification = None
            self.current_color = None
            
            # Store frames for each category for GUI display
            self.last_proper_frame = None
            self.last_unperforated_frame = None  
            self.last_wrong_frame = None
        except Exception as e:
            self.loggerReport.logger.error("Error initializing RunCamera: " + str(e))

    def start_camera(self):
        try:
            self.stream = cv2.VideoCapture(self.src)
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
        
        # Resize frame to match des_parcial.py processing size for consistent area calculations
        processed_frame = cv2.resize(processed_frame, (1280, 720))
        frame_height, frame_width = processed_frame.shape[:2]

        hsv_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        
        # Get the best detection mask using the classifier
        detection_mask = classifier.get_best_detection_mask(hsv_frame)
        
        # Find contours in the detection mask
        contours, _ = cv2.findContours(detection_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            # Find the largest contour for piece detection
            contour_areas = [cv2.contourArea(item) for item in contours]
            max_area = max(contour_areas)
            
            if max_area > 3000:  # Lower threshold for better detection
                max_area_contour = contours[contour_areas.index(max_area)]
                x, y, w, h = cv2.boundingRect(max_area_contour)
                
                # Calculate the center of the bounding box
                center_x = x + w // 2
                
                # Use consistent center detection logic as des_parcial.py
                detection_zone_center = 640  # Fixed center for 1280 width
                detection_tolerance = 50
                
                # Detect when a piece is at the center of the screen (around pixel 640)
                # Allow some tolerance (±50 pixels) around the center  
                if (590 <= center_x <= 690 and self.testigo and max_area > 3000):
                    
                    # Create a region of interest around the detected piece for classification
                    roi_x1 = max(0, x - 20)
                    roi_y1 = max(0, y - 20)
                    roi_x2 = min(frame_width, x + w + 20)
                    roi_y2 = min(frame_height, y + h + 20)
                    
                    roi_hsv = hsv_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    
                    # Draw bounding rectangle on the processed frame
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Draw center line for reference
                    cv2.line(processed_frame, (640, 0), (640, frame_height), (255, 0, 0), 2)
                    # Draw center point of the detected object
                    cv2.circle(processed_frame, (center_x, y + h // 2), 5, (0, 0, 255), -1)
                    
                    # Classify the piece using the ROI
                    classification, detected_color = classifier.classify_piece(roi_hsv)
                    
                    # Store current classification for GUI display
                    self.current_classification = classification
                    self.current_color = detected_color
                    
                    # Update counters based on classification
                    if classification == 'unperforated':
                        self.cont_unperforated += 1
                        print(f"Unperforated piece detected! Count: {self.cont_unperforated} (Color: {detected_color})")
                        # Store frame for GUI display
                        self.last_unperforated_frame = processed_frame.copy()
                    elif classification == 'wrong':
                        self.cont_wrong += 1
                        print(f"Wrong piece detected! Count: {self.cont_wrong} (Color: {detected_color})")
                        # Store frame for GUI display
                        self.last_wrong_frame = processed_frame.copy()
                    elif classification == 'proper':
                        self.cont_proper += 1
                        print(f"Proper piece detected! Count: {self.cont_proper} (Color: {detected_color})")
                        # Store frame for GUI display
                        self.last_proper_frame = processed_frame.copy()
                    
                    # Store the processed frame when detection occurs
                    self.processed_frame = processed_frame.copy()
                    self.detection_occurred = True
                    
                    self.testigo = False
                
                # Reset testigo when object moves away from center
                elif center_x < 540 or center_x > 740:
                    self.testigo = True
