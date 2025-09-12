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
        
        self.master = master
        self.width = 1280
        self.height = 720
        self.master.geometry("%dx%d" % (self.width, self.height))
        
        # Initialize camera-related data structures
        self.camera_real_time = None
        self.camera_detected = None
        self.label_real_time = None
        self.label_detected = None
        self.frame_real_time = None
        self.frame_detected = None
        self.imgTk_real_time = None
        self.imgTk_detected = None
        
        # Counter variables
        self.pieza_1_count = 0
        self.pieza_2_count = 0
        
        #Create widgets
        self.create_widgets()
        self.createFrames()
        
        self.master.mainloop()
        
    def createFrames(self):
        # Create video display boxes for the two cameras
        video_width = 320
        video_height = 240
        
        # Real-time footage box (left)
        self.label_real_time = tk.Label(
            self.master,
            borderwidth=2,
            relief="solid",
            width=video_width,
            height=video_height
        )
        self.label_real_time.place(x=50, y=80)
        
        # Detected object box (right)
        self.label_detected = tk.Label(
            self.master,
            borderwidth=2,
            relief="solid",
            width=video_width,
            height=video_height
        )
        self.label_detected.place(x=450, y=80)
        
        # Create initial black images
        self.createImageZeros("real_time")
        self.createImageZeros("detected")
        
        # Configure images for both labels
        self.label_real_time.configure(image=self.imgTk_real_time)
        self.label_real_time.image = self.imgTk_real_time
        
        self.label_detected.configure(image=self.imgTk_detected)
        self.label_detected.image = self.imgTk_detected

    def createImageZeros(self, camera_type):
        frame = np.zeros([240, 320, 3], dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(frame)
        
        if camera_type == "real_time":
            self.imgTk_real_time = ImageTk.PhotoImage(image=imgArray)
            self.frame_real_time = frame
        elif camera_type == "detected":
            self.imgTk_detected = ImageTk.PhotoImage(image=imgArray)
            self.frame_detected = frame

    def create_widgets(self):
        self.fontLabelText = font.Font(
            family = "Helvetica",
            size = 16,
            weight = "normal"
        )
        
        self.fontCounterText = font.Font(
            family = "Helvetica",
            size = 14,
            weight = "bold"
        )
        
        # Camera labels below video boxes
        self.label_real_time_text = tk.Label(
            self.master, 
            text = "Real time footage",
            fg = "#000000"
        )
        self.label_real_time_text['font'] = self.fontLabelText
        self.label_real_time_text.place(x = 120, y = 330)
        
        self.label_detected_text = tk.Label(
            self.master, 
            text = "Detected Object",
            fg = "#000000"
        )
        self.label_detected_text['font'] = self.fontLabelText
        self.label_detected_text.place(x = 490, y = 330)

        # Counters section
        counter_start_x = 800
        counter_start_y = 100
        
        # Pieza_1 counter
        self.label_pieza_1 = tk.Label(
            self.master,
            text = "Pieza_1 (O):",
            fg = "#000000"
        )
        self.label_pieza_1['font'] = self.fontCounterText
        self.label_pieza_1.place(x = counter_start_x, y = counter_start_y)
        
        self.label_pieza_1_count = tk.Label(
            self.master,
            text = str(self.pieza_1_count),
            fg = "#000000",
            bg = "#f0f0f0",
            borderwidth=2,
            relief="solid",
            width = 10
        )
        self.label_pieza_1_count['font'] = self.fontCounterText
        self.label_pieza_1_count.place(x = counter_start_x + 150, y = counter_start_y)
        
        # Pieza_2 counter
        self.label_pieza_2 = tk.Label(
            self.master,
            text = "Pieza_2 (8):",
            fg = "#000000"
        )
        self.label_pieza_2['font'] = self.fontCounterText
        self.label_pieza_2.place(x = counter_start_x, y = counter_start_y + 50)
        
        self.label_pieza_2_count = tk.Label(
            self.master,
            text = str(self.pieza_2_count),
            fg = "#000000",
            bg = "#f0f0f0",
            borderwidth=2,
            relief="solid",
            width = 10
        )
        self.label_pieza_2_count['font'] = self.fontCounterText
        self.label_pieza_2_count.place(x = counter_start_x + 150, y = counter_start_y + 50)

        # Create control buttons
        button_y = 500
        self.btnInitCamera = tk.Button(
            self.master,
            text="Init Real-time Camera",
            bg = "#0aa50a",
            fg = "#ffffff",
            width = 18,
            command=self.init_real_time_camera
        )
        self.btnInitCamera.place(x = 50, y = button_y)

        self.btnStopCamera = tk.Button(
            self.master,
            text="Stop Real-time Camera",
            bg = "#aa0a0a",
            fg = "#ffffff",
            width = 18,
            command=self.stop_real_time_camera
        )
        self.btnStopCamera.place(x = 50, y = button_y + 40)
        
        # Reset counters button
        self.btnResetCounters = tk.Button(
            self.master,
            text="Reset Counters",
            bg = "#0066cc",
            fg = "#ffffff",
            width = 15,
            command=self.reset_counters
        )
        self.btnResetCounters.place(x = counter_start_x, y = counter_start_y + 120)

    def init_real_time_camera(self):
        print("Initializing real-time camera...")
        # Reset the label text in case it was showing "VIDEO ENDED"
        self.label_real_time_text.configure(text="Real time footage", fg="#000000")
        
        # Reset counters
        self.pieza_1_count = 0
        self.pieza_2_count = 0
        self.label_pieza_1_count.configure(text=str(self.pieza_1_count))
        self.label_pieza_2_count.configure(text=str(self.pieza_2_count))
        
        self.camera_real_time = camera.RunCamera(src=r'imgs\video_1_12.avi', name="RealTimeCamera")
        self.camera_real_time.start_camera()
        self.show_real_time_video()

    def show_real_time_video(self):
        if self.camera_real_time and self.camera_real_time.frame is not None and self.camera_real_time.is_playing():
            imgTk = self.convertToFrameTk(self.camera_real_time.frame)
            self.label_real_time.configure(image=imgTk)
            self.label_real_time.image = imgTk
            
            # Update counters from camera
            self.update_counters_from_camera()
            
            # Check for new detections and update detected object display
            self.update_detected_display()
            
            self.label_real_time.after(30, self.show_real_time_video)
        elif self.camera_real_time and not self.camera_real_time.is_playing():
            # Video has ended, print to console and GUI
            print("VIDEO PLAYBACK COMPLETED!")
            print(f"Video '{self.camera_real_time.src}' has finished playing.")
            print(f"Final counts - Pieza_1: {self.camera_real_time.cont1}, Pieza_2: {self.camera_real_time.cont2}")
            
            # Optionally update GUI to show video ended
            self.label_real_time_text.configure(text="Real time footage - VIDEO ENDED", fg="#ff0000")
        else:
            # Continue checking
            self.label_real_time.after(30, self.show_real_time_video)


    def update_detected_display(self):
        """Check for new detections and update the detected object display"""
        if self.camera_real_time and self.camera_real_time.detection_occurred:
            # Display the processed frame with detection rectangles
            if self.camera_real_time.processed_frame is not None:
                imgTk = self.convertToFrameTk(self.camera_real_time.processed_frame)
                self.label_detected.configure(image=imgTk)
                self.label_detected.image = imgTk
                
                # Reset the detection flag
                self.camera_real_time.detection_occurred = False

    def update_counters_from_camera(self):
        """Update GUI counters with values from camera object detection"""
        if self.camera_real_time:
            # Update Pieza_1 counter from camera's cont1
            if self.pieza_1_count != self.camera_real_time.cont1:
                self.pieza_1_count = self.camera_real_time.cont1
                self.label_pieza_1_count.configure(text=str(self.pieza_1_count))
                print(f"Updated Pieza_1 counter to: {self.pieza_1_count}")
            
            # Update Pieza_2 counter from camera's cont2
            if self.pieza_2_count != self.camera_real_time.cont2:
                self.pieza_2_count = self.camera_real_time.cont2
                self.label_pieza_2_count.configure(text=str(self.pieza_2_count))
                print(f"Updated Pieza_2 counter to: {self.pieza_2_count}")

    def convertToFrameTk(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame to fit the display area
        frame = cv2.resize(frame, (320, 240))
        imgArray = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=imgArray)

    def stop_real_time_camera(self):
        print("Stopping real-time camera...")
        if self.camera_real_time:
            self.camera_real_time.stop()
            self.camera_real_time = None

    def update_pieza_1_count(self, count=None):
        """Update Pieza_1 counter. If count is None, increment by 1"""
        if count is not None:
            self.pieza_1_count = count
        else:
            self.pieza_1_count += 1
        self.label_pieza_1_count.configure(text=str(self.pieza_1_count))

    def update_pieza_2_count(self, count=None):
        """Update Pieza_2 counter. If count is None, increment by 1"""
        if count is not None:
            self.pieza_2_count = count
        else:
            self.pieza_2_count += 1
        self.label_pieza_2_count.configure(text=str(self.pieza_2_count))

    def reset_counters(self):
        """Reset both counters to zero"""
        # Reset GUI counters
        self.pieza_1_count = 0
        self.pieza_2_count = 0
        self.label_pieza_1_count.configure(text=str(self.pieza_1_count))
        self.label_pieza_2_count.configure(text=str(self.pieza_2_count))
        
        # Reset camera counters if camera is active
        if self.camera_real_time:
            self.camera_real_time.cont1 = 0
            self.camera_real_time.cont2 = 0
            print("Counters reset to zero")


def main():
    root = tk.Tk()
    root.title("GUI Camera - Object Detection")
    appRunCamera = Application(master=root)
