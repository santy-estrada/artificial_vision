from logger import Logger
import tkinter as tk
#pip install pillow
from PIL import Image, ImageTk
import tkinter.font as font
from tkinter import ttk
#pip install opencv-python
import cv2
import camera
import numpy as np

class Application(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.logReport = Logger("logGui")
        self.logReport.logger.info("Init GUI constructor Application")
        self.frame = None
        self.imgTk = None

        self.master = master
        self.width = 1080
        self.height = 720
        self.master.geometry("%dx%d" % (self.width, self.height))
        
        #Create widgets
        self.create_widgets()
        self.createFrame()
        
        self.master.mainloop()
        
    def createFrame(self):
        self.labelVideo_1 = tk.Label(
            self.master,
            borderwidth=2,
            relief="solid"
        )
        self.labelVideo_1.place(x=10, y=60)
        self.createImageZeros() 
        self.labelVideo_1.configure(image=self.imgTk)
        self.labelVideo_1.image = self.imgTk

    def createImageZeros(self):
        self.frame = np.zeros([480, 320, 3], dtype=np.uint8)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(self.frame)
        self.imgTk = ImageTk.PhotoImage(image=imgArray)

    def create_widgets(self):
        self.fontLabelText = font.Font(
            family = "Helvetica",
            size = 20,
            weight = "normal"
        )
        self.labelNameCamera = tk.Label(
            self.master, 
            text = "Camera 1",
            fg = "#000000"
        )
        self.labelNameCamera['font'] = self.fontLabelText
        self.labelNameCamera.place(x = 10, y = 10)

        self.btnInitCamera = tk.Button(
            self.master,
            text="Init Camera",
            bg = "#0aa50a",
            fg = "#ffffff",
            width = 12,
            command=self.init_camera
        )
        self.btnInitCamera.place(x = 100, y = 600)

        self.btnStopCamera = tk.Button(
            self.master,
            text="Stop Camera",
            bg = "#aa0a0a",
            fg = "#ffffff",
            width = 12,
            command=self.stop_camera
        )
        self.btnStopCamera.place(x = 220, y = 600)

    def init_camera(self):
        print("Initializing camera...")
        self.camera_1 = camera.RunCamera(src = 0, name = "MyCamera1")
        self.camera_1.start_camera()
        self.show_Video()

    def show_Video(self):
        if (self.camera_1.frame is not None):
            imgTk = self.convertToFrameTk(self.camera_1.frame)
            self.labelVideo_1.configure(image=imgTk)
            self.labelVideo_1.image = imgTk
        self.labelVideo_1.after(1, self.show_Video)

    def convertToFrameTk(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=imgArray)
        

    def stop_camera(self):
        print("Stopping camera...")
        if self.camera_1:
            self.camera_1.stop()


def main():
    root = tk.Tk()
    root.title("GUI Camera")
    appRunCamera = Application(master=root)
