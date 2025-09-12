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
        self.width = 1500  # Increased width further to accommodate moved counters
        self.height = 900  # Increased height for 4 video windows
        self.master.geometry("%dx%d" % (self.width, self.height))
        self.master.title("Advanced Piece Classification System")
        
        # Initialize camera-related data structures for 4 windows
        self.camera_real_time = None
        
        # Labels for 4 video windows
        self.label_main_video = None      # Upper left - main video
        self.label_proper_video = None    # Upper right - proper pieces
        self.label_unperforated_video = None  # Lower left - unperforated pieces  
        self.label_wrong_video = None     # Lower right - wrong pieces
        
        # Image variables for 4 windows
        self.imgTk_main = None
        self.imgTk_proper = None
        self.imgTk_unperforated = None
        self.imgTk_wrong = None
        
        # Store last detected frames for each category
        self.last_proper_frame = None
        self.last_unperforated_frame = None
        self.last_wrong_frame = None
        
        # Counter variables
        self.unperforated_count = 0
        self.wrong_count = 0
        self.proper_count = 0
        self.total_count = 0  # Total pieces counter
        
        #Create widgets
        self.create_widgets()
        self.createFrames()
        
        self.master.mainloop()
        
    def createFrames(self):
        # Create 4 video display windows
        video_width = 300
        video_height = 225
        margin_x = 20
        margin_y = 70  # Moved down from 20 to 70 to accommodate title
        spacing_x = 350  # Space between columns
        spacing_y = 275  # Space between rows
        
        # Upper left - Main video feed
        self.label_main_video = tk.Label(
            self.master,
            borderwidth=2,
            relief="solid",
            width=video_width,
            height=video_height,
            bg="black"
        )
        self.label_main_video.place(x=margin_x, y=margin_y)
        
        # Upper right - Proper pieces
        self.label_proper_video = tk.Label(
            self.master,
            borderwidth=2,
            relief="solid",
            width=video_width,
            height=video_height,
            bg="lightgreen"
        )
        self.label_proper_video.place(x=margin_x + spacing_x, y=margin_y)
        
        # Lower left - Unperforated pieces
        self.label_unperforated_video = tk.Label(
            self.master,
            borderwidth=2,
            relief="solid",
            width=video_width,
            height=video_height,
            bg="lightcoral"
        )
        self.label_unperforated_video.place(x=margin_x, y=margin_y + spacing_y)
        
        # Lower right - Wrong pieces
        self.label_wrong_video = tk.Label(
            self.master,
            borderwidth=2,
            relief="solid",
            width=video_width,
            height=video_height,
            bg="lightyellow"
        )
        self.label_wrong_video.place(x=margin_x + spacing_x, y=margin_y + spacing_y)
        
        # Create initial black images for all windows
        self.createImageZeros("main")
        self.createImageZeros("proper")
        self.createImageZeros("unperforated")
        self.createImageZeros("wrong")
        
        # Configure images for all labels
        self.label_main_video.configure(image=self.imgTk_main)
        self.label_main_video.image = self.imgTk_main
        
        self.label_proper_video.configure(image=self.imgTk_proper)
        self.label_proper_video.image = self.imgTk_proper
        
        self.label_unperforated_video.configure(image=self.imgTk_unperforated)
        self.label_unperforated_video.image = self.imgTk_unperforated
        
        self.label_wrong_video.configure(image=self.imgTk_wrong)
        self.label_wrong_video.image = self.imgTk_wrong

    def createImageZeros(self, window_type):
        frame = np.zeros([225, 300, 3], dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(frame)
        
        if window_type == "main":
            self.imgTk_main = ImageTk.PhotoImage(image=imgArray)
        elif window_type == "proper":
            self.imgTk_proper = ImageTk.PhotoImage(image=imgArray)
        elif window_type == "unperforated":
            self.imgTk_unperforated = ImageTk.PhotoImage(image=imgArray)
        elif window_type == "wrong":
            self.imgTk_wrong = ImageTk.PhotoImage(image=imgArray)

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
        
        # Title font with larger size
        self.fontTitleText = font.Font(
            family = "Helvetica",
            size = 18,
            weight = "bold"
        )
        
        # Main title at the top
        self.label_title = tk.Label(
            self.master, 
            text = "Advanced Piece Classification System",
            fg = "#000080"  # Dark blue color for the title
        )
        self.label_title['font'] = self.fontTitleText
        self.label_title.place(x = 450, y = 10)  # Centered near the top

        # Video window labels
        label_y = 310  # Below first row of videos (moved down 50px from 260)
        
        self.label_main_text = tk.Label(
            self.master, 
            text = "Main Video Feed",
            fg = "#000000"
        )
        self.label_main_text['font'] = self.fontLabelText
        self.label_main_text.place(x = 100, y = label_y)
        
        self.label_proper_text = tk.Label(
            self.master, 
            text = "Proper Pieces",
            fg = "#006600"
        )
        self.label_proper_text['font'] = self.fontLabelText
        self.label_proper_text.place(x = 450, y = label_y)
        
        # Labels for bottom row
        label_y2 = label_y + 275  # Below second row of videos
        
        self.label_unperforated_text = tk.Label(
            self.master, 
            text = "Unperforated Pieces",
            fg = "#cc0000"
        )
        self.label_unperforated_text['font'] = self.fontLabelText
        self.label_unperforated_text.place(x = 70, y = label_y2)
        
        self.label_wrong_text = tk.Label(
            self.master, 
            text = "Wrong Pieces",
            fg = "#cc6600"
        )
        self.label_wrong_text['font'] = self.fontLabelText
        self.label_wrong_text.place(x = 450, y = label_y2)

        # Counters section - move further to the right
        counter_start_x = 750  # Moved further right from 750
        counter_start_y = 50
        
        # Total pieces counter (add at top)
        self.label_total = tk.Label(
            self.master,
            text = "Total Pieces:",
            fg = "#000000"
        )
        self.label_total['font'] = self.fontCounterText
        self.label_total.place(x = counter_start_x, y = counter_start_y)
        
        self.label_total_count = tk.Label(
            self.master,
            text = str(self.total_count),
            fg = "#000000",
            bg = "#e6e6e6",  # Light gray background
            borderwidth=2,
            relief="solid",
            width = 10
        )
        self.label_total_count['font'] = self.fontCounterText
        self.label_total_count.place(x = counter_start_x + 180, y = counter_start_y)
        
        # Unperforated pieces counter
        self.label_unperforated = tk.Label(
            self.master,
            text = "Unperforated:",
            fg = "#000000"
        )
        self.label_unperforated['font'] = self.fontCounterText
        self.label_unperforated.place(x = counter_start_x, y = counter_start_y + 40)
        
        self.label_unperforated_count = tk.Label(
            self.master,
            text = str(self.unperforated_count),
            fg = "#000000",
            bg = "#ffdddd",  # Light red background
            borderwidth=2,
            relief="solid",
            width = 10
        )
        self.label_unperforated_count['font'] = self.fontCounterText
        self.label_unperforated_count.place(x = counter_start_x + 180, y = counter_start_y + 40)
        
        # Wrong pieces counter
        self.label_wrong = tk.Label(
            self.master,
            text = "Wrong:",
            fg = "#000000"
        )
        self.label_wrong['font'] = self.fontCounterText
        self.label_wrong.place(x = counter_start_x, y = counter_start_y + 80)
        
        self.label_wrong_count = tk.Label(
            self.master,
            text = str(self.wrong_count),
            fg = "#000000",
            bg = "#ffffdd",  # Light yellow background
            borderwidth=2,
            relief="solid",
            width = 10
        )
        self.label_wrong_count['font'] = self.fontCounterText
        self.label_wrong_count.place(x = counter_start_x + 180, y = counter_start_y + 80)
        
        # Proper pieces counter
        self.label_proper = tk.Label(
            self.master,
            text = "Proper:",
            fg = "#000000"
        )
        self.label_proper['font'] = self.fontCounterText
        self.label_proper.place(x = counter_start_x, y = counter_start_y + 120)
        
        self.label_proper_count = tk.Label(
            self.master,
            text = str(self.proper_count),
            fg = "#000000",
            bg = "#ddffdd",  # Light green background
            borderwidth=2,
            relief="solid",
            width = 10
        )
        self.label_proper_count['font'] = self.fontCounterText
        self.label_proper_count.place(x = counter_start_x + 180, y = counter_start_y + 120)

        # Classification info display - larger and moved down
        self.label_current_class = tk.Label(
            self.master,
            text = "Last Classification:",
            fg = "#000000"
        )
        self.label_current_class['font'] = self.fontCounterText
        self.label_current_class.place(x = counter_start_x, y = counter_start_y + 170)
        
        self.label_current_class_value = tk.Label(
            self.master,
            text = "None",
            fg = "#0066cc",
            bg = "#f0f0f0",
            borderwidth=2,
            relief="solid",
            width = 25,  # Increased width from 15 to 25
            height = 2   # Added height for larger text box
        )
        self.label_current_class_value['font'] = self.fontCounterText
        self.label_current_class_value.place(x = counter_start_x, y = counter_start_y + 200)

        # Create control buttons - move to bottom
        button_y = 650
        self.btnInitCamera = tk.Button(
            self.master,
            text="Start Classification",
            bg = "#0aa50a",
            fg = "#ffffff",
            width = 18,
            command=self.init_real_time_camera
        )
        self.btnInitCamera.place(x = 50, y = button_y)

        self.btnStopCamera = tk.Button(
            self.master,
            text="Stop Classification",
            bg = "#aa0a0a",
            fg = "#ffffff",
            width = 18,
            command=self.stop_real_time_camera
        )
        self.btnStopCamera.place(x = 50, y = button_y + 40)
        
        # Reset counters button - moved down to accommodate larger classification display
        self.btnResetCounters = tk.Button(
            self.master,
            text="Reset Counters",
            bg = "#0066cc",
            fg = "#ffffff",
            width = 15,
            command=self.reset_counters
        )
        self.btnResetCounters.place(x = counter_start_x, y = counter_start_y + 260)

    def init_real_time_camera(self):
        print("Initializing real-time camera...")
        # Reset the label text in case it was showing "VIDEO ENDED"
        self.label_main_text.configure(text="Main Video Feed", fg="#000000")
        
        # Reset counters
        self.unperforated_count = 0
        self.wrong_count = 0
        self.proper_count = 0
        self.total_count = 0  # Reset total counter
        self.label_unperforated_count.configure(text=str(self.unperforated_count))
        self.label_wrong_count.configure(text=str(self.wrong_count))
        self.label_proper_count.configure(text=str(self.proper_count))
        self.label_total_count.configure(text=str(self.total_count))
        self.label_current_class_value.configure(text="None")
        
        # Reset stored frames
        self.last_proper_frame = None
        self.last_unperforated_frame = None
        self.last_wrong_frame = None
        
        self.camera_real_time = camera.RunCamera(src=r'imgs\video_3.mp4', name="RealTimeCamera")
        self.camera_real_time.start_camera()
        self.show_real_time_video()

    def show_real_time_video(self):
        if self.camera_real_time and self.camera_real_time.frame is not None and self.camera_real_time.is_playing():
            # Update main video window (upper left)
            imgTk = self.convertToFrameTk(self.camera_real_time.frame, (300, 225))
            self.label_main_video.configure(image=imgTk)
            self.label_main_video.image = imgTk
            
            # Update counters from camera
            self.update_counters_from_camera()
            
            # Check for new detections and update category-specific displays
            self.update_category_displays()
            
            
        elif self.camera_real_time and not self.camera_real_time.is_playing():
            # Video has ended, print to console and GUI
            print("VIDEO PLAYBACK COMPLETED!")
            print(f"Video '{self.camera_real_time.src}' has finished playing.")
            print(f"Final counts - Unperforated: {self.camera_real_time.cont_unperforated}, Wrong: {self.camera_real_time.cont_wrong}, Proper: {self.camera_real_time.cont_proper}")
            
            # Optionally update GUI to show video ended
            self.label_main_text.configure(text="Main Video Feed - VIDEO ENDED", fg="#ff0000")

        self.label_main_video.after(1, self.show_real_time_video)


    def update_category_displays(self):
        """Update the category-specific video displays when new detections occur"""
        if self.camera_real_time and self.camera_real_time.detection_occurred:
            if (self.camera_real_time.processed_frame is not None and 
                self.camera_real_time.current_classification):
                
                # Store the frame based on classification
                classification = self.camera_real_time.current_classification
                
                if classification == 'proper':
                    self.last_proper_frame = self.camera_real_time.processed_frame.copy()
                    imgTk = self.convertToFrameTk(self.last_proper_frame, (300, 225))
                    self.label_proper_video.configure(image=imgTk)
                    self.label_proper_video.image = imgTk
                    
                elif classification == 'unperforated':
                    self.last_unperforated_frame = self.camera_real_time.processed_frame.copy()
                    imgTk = self.convertToFrameTk(self.last_unperforated_frame, (300, 225))
                    self.label_unperforated_video.configure(image=imgTk)
                    self.label_unperforated_video.image = imgTk
                    
                elif classification == 'wrong':
                    self.last_wrong_frame = self.camera_real_time.processed_frame.copy()
                    imgTk = self.convertToFrameTk(self.last_wrong_frame, (300, 225))
                    self.label_wrong_video.configure(image=imgTk)
                    self.label_wrong_video.image = imgTk
                
                # Reset the detection flag
                self.camera_real_time.detection_occurred = False

    def update_counters_from_camera(self):
        """Update GUI counters with values from camera object detection"""
        if self.camera_real_time:
            # Update Unperforated counter
            if self.unperforated_count != self.camera_real_time.cont_unperforated:
                self.unperforated_count = self.camera_real_time.cont_unperforated
                self.label_unperforated_count.configure(text=str(self.unperforated_count))
                print(f"Updated Unperforated counter to: {self.unperforated_count}")
            
            # Update Wrong counter
            if self.wrong_count != self.camera_real_time.cont_wrong:
                self.wrong_count = self.camera_real_time.cont_wrong
                self.label_wrong_count.configure(text=str(self.wrong_count))
                print(f"Updated Wrong counter to: {self.wrong_count}")
                
            # Update Proper counter
            if self.proper_count != self.camera_real_time.cont_proper:
                self.proper_count = self.camera_real_time.cont_proper
                self.label_proper_count.configure(text=str(self.proper_count))
                print(f"Updated Proper counter to: {self.proper_count}")
                
            # Update Total counter
            new_total = self.unperforated_count + self.wrong_count + self.proper_count
            if self.total_count != new_total:
                self.total_count = new_total
                self.label_total_count.configure(text=str(self.total_count))
                print(f"Updated Total counter to: {self.total_count}")
                
            # Update current classification display
            if (self.camera_real_time.current_classification and 
                self.camera_real_time.current_color):
                classification_text = f"{self.camera_real_time.current_classification} ({self.camera_real_time.current_color})"
                self.label_current_class_value.configure(text=classification_text)
                
                # Update color based on classification
                if self.camera_real_time.current_classification == 'unperforated':
                    self.label_current_class_value.configure(bg="#ffdddd")  # Light red
                elif self.camera_real_time.current_classification == 'wrong':
                    self.label_current_class_value.configure(bg="#ffffdd")  # Light yellow
                elif self.camera_real_time.current_classification == 'proper':
                    self.label_current_class_value.configure(bg="#ddffdd")  # Light green

    def convertToFrameTk(self, frame, size=(300, 225)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame to fit the display area
        frame = cv2.resize(frame, size)
        imgArray = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=imgArray)

    def stop_real_time_camera(self):
        print("Stopping real-time camera...")
        if self.camera_real_time:
            self.camera_real_time.stop()
            self.camera_real_time = None

    def reset_counters(self):
        """Reset all counters to zero"""
        # Reset GUI counters
        self.unperforated_count = 0
        self.wrong_count = 0
        self.proper_count = 0
        self.total_count = 0  # Reset total counter
        self.label_unperforated_count.configure(text=str(self.unperforated_count))
        self.label_wrong_count.configure(text=str(self.wrong_count))
        self.label_proper_count.configure(text=str(self.proper_count))
        self.label_total_count.configure(text=str(self.total_count))
        self.label_current_class_value.configure(text="None", bg="#f0f0f0")
        
        # Reset stored frames
        self.last_proper_frame = None
        self.last_unperforated_frame = None
        self.last_wrong_frame = None
        
        # Clear category video displays
        self.createImageZeros("proper")
        self.createImageZeros("unperforated") 
        self.createImageZeros("wrong")
        
        self.label_proper_video.configure(image=self.imgTk_proper)
        self.label_proper_video.image = self.imgTk_proper
        
        self.label_unperforated_video.configure(image=self.imgTk_unperforated)
        self.label_unperforated_video.image = self.imgTk_unperforated
        
        self.label_wrong_video.configure(image=self.imgTk_wrong)
        self.label_wrong_video.image = self.imgTk_wrong
        
        # Reset camera counters if camera is active
        if self.camera_real_time:
            self.camera_real_time.cont_unperforated = 0
            self.camera_real_time.cont_wrong = 0
            self.camera_real_time.cont_proper = 0
            self.camera_real_time.current_classification = None
            self.camera_real_time.current_color = None
            print("Counters reset to zero")


def main():
    root = tk.Tk()
    root.title("GUI Camera - Object Detection")
    appRunCamera = Application(master=root)
