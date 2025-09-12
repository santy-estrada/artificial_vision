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
            self.stopped = False
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

    def get(self):
        while not self.stopped:
            if not self.ret:
                pass
            else:
                try:
                    self.ret, self.frame = self.stream.read()
                except Exception as e:
                    self.loggerReport.logger.error("Error in get Camera: " + str(e))
           