from logger import Logger
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.font as font
from tkinter import ttk
import cv2
import camera
import numpy as np


class Application(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.logReport = Logger("logGui")
        self.logReport.logger.info("init GUI")
        
        self.master = master
        self.width = 1280
        self.height = 720
        self.master.geometry("%dx%d" % (self.width, self.height))
        self.is_camera_running = False
        
        self.createWidgets()
        self.createFrames()
        
        self.master.mainloop()

    def createFrames(self):
        # Frame para el video principal
        self.labelVideo_1 = tk.Label(self.master, borderwidth=2, relief="solid")
        self.labelVideo_1.place(x=10, y=50)
                
        self.labelVideo_placaDetectada = tk.Label(self.master, borderwidth=2, relief="solid")
        self.labelVideo_placaDetectada.place(x=750, y=50)

        self.imgTk_placeholder = self.createImagePlaceholder(720, 480)
        self.labelVideo_1.configure(image=self.imgTk_placeholder)
        self.labelVideo_1.image = self.imgTk_placeholder

        self.imgTk_placa_detectada_placeholder = self.createImagePlaceholder(round(320*1.5), round(200*1.5))
        self.labelVideo_placaDetectada.configure(image=self.imgTk_placa_detectada_placeholder)
        self.labelVideo_placaDetectada.image = self.imgTk_placa_detectada_placeholder

    def createImagePlaceholder(self, width, height):
        frame = np.zeros([height, width, 3], dtype=np.uint8)
        img_array = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=img_array)

    def createWidgets(self):
        self.fontLabelText = font.Font(family='Helvetica', size=10, weight='bold')
        self.fontCounters = font.Font(family='Helvetica', size=12, weight='bold')
        
        tk.Label(self.master, text="Cámara Principal", font=self.fontLabelText).place(x=10, y=20)
        tk.Label(self.master, text="Placa detectada", font=self.fontLabelText).place(x=750, y=20)

        # Placa detectada
        self.placa_detectada_var = tk.StringVar(value="Placa detectada: ")
        tk.Label(self.master, textvariable=self.placa_detectada_var, font=self.fontCounters, fg="black").place(x=750, y=380)

        # Total de carros
        self.total_carros_var = tk.StringVar(value="Total de carros detectados: ")
        tk.Label(self.master, textvariable=self.total_carros_var, font=self.fontCounters, fg="black").place(x=750, y=480)

        # Botones
        self.btnInitCamera = tk.Button(self.master, text='Iniciar Cámara', bg="#06542a", fg='#ffffff', width=15, command=self.initCamera)
        self.btnInitCamera.place(x=50, y=600)

        self.btnStopCamera = tk.Button(self.master, text='Parar Cámara', bg="#511610", fg='#ffffff', width=15, command=self.stopCamera, state=tk.DISABLED)
        self.btnStopCamera.place(x=250, y=600)

    def initCamera(self):
        self.logReport.logger.info("Iniciando video...")
        
        video_path = r"C:\Personal\Samuel\UNIVERSIDAD\Decimo_semestre\VIsion_artificial\Proyecto_deteccion_placas\Videos\Video_deteccion.mp4"
        self.camera_1 = camera.RunCamera(src=video_path, name="Video portería")
        self.camera_1.start()
        self.is_camera_running = True
        self.btnInitCamera.config(state=tk.DISABLED)
        self.btnStopCamera.config(state=tk.NORMAL)
        
        self.showVideo()
        

    def showVideo(self):
        if not self.is_camera_running:
            return

        # Actualizar el frame de la cámara principal
        if self.camera_1.video_entrada is not None:
            frame_resized = cv2.resize(self.camera_1.video_entrada, (720, 480))
            imgTk = self.convertToFrameTk(frame_resized)
            self.labelVideo_1.configure(image=imgTk)
            self.labelVideo_1.image = imgTk

        if self.camera_1.placa_detectada is not None:
            placa_detectada_mask_resized = cv2.resize(self.camera_1.placa_detectada, (round(320*1.5), round(200*1.5)))
            imgTk_placa_detectada = self.convertToFrameTk(placa_detectada_mask_resized)
            self.labelVideo_placaDetectada.configure(image=imgTk_placa_detectada)
            self.labelVideo_placaDetectada.image = imgTk_placa_detectada
        else:
            self.labelVideo_placaDetectada.configure(image=self.imgTk_placa_detectada_placeholder)
            self.labelVideo_placaDetectada.image = self.imgTk_placa_detectada_placeholder


        self.placa_detectada_var.set(f"Placa detectada: {self.camera_1.placa_detectada_var}")
        self.total_carros_var.set(f"Total de carros detectados: {self.camera_1.total_carros_var}")



        self.labelVideo_1.after(10, self.showVideo)

    def convertToFrameTk(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_array = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=img_array)

    def stopCamera(self):
        self.logReport.logger.info("Deteniendo cámara...")
        if hasattr(self, 'camera_1') and self.camera_1:
            self.camera_1.stop()
        
        self.is_camera_running = False
        self.btnInitCamera.config(state=tk.NORMAL)
        self.btnStopCamera.config(state=tk.DISABLED)

        self.labelVideo_1.configure(image=self.imgTk_placeholder)
        self.labelVideo_1.image = self.imgTk_placeholder

        self.labelVideo_placaDetectada.configure(image=self.imgTk_placa_detectada_placeholder)
        self.labelVideo_placaDetectada.image = self.imgTk_placa_detectada_placeholder
        
def main():
    root = tk.Tk()
    root.title("Sistema de detección de placas")
    appRunCamera = Application(master=root)

if __name__ == '__main__':
    main()