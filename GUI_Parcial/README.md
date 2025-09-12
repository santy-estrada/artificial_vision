# Advanced Piece Classification GUI System

## Overview
This GUI application provides real-time piece classification using computer vision techniques. It analyzes video feeds to detect and classify pieces into three categories: Proper, Unperforated, and Wrong pieces.

## Features
- **4 Video Windows Layout**: Main feed plus dedicated windows for each piece category
- **Advanced Color Detection**: Supports Green, Blue, Yellow, Red, and Purple pieces
- **Real-time Classification**: Processes video frames and classifies pieces instantly
- **Smart Filtering**: Uses exclusion logic to handle overlapping color ranges
- **Live Counters**: Tracks count of each piece category
- **Visual Feedback**: Color-coded displays for easy identification

## GUI Layout

### Video Windows (2x2 Grid)
1. **Upper Left - Main Video Feed**
   - Live video processing with detection rectangles
   - Shows bounding boxes around detected pieces
   - Real-time frame processing

2. **Upper Right - Proper Pieces** 
   - Displays last detected proper piece
   - Green-tinted background
   - Updates when 2-hole pieces are classified

3. **Lower Left - Unperforated Pieces**
   - Shows last detected unperforated piece  
   - Red-tinted background
   - Updates when solid pieces are classified

4. **Lower Right - Wrong Pieces**
   - Displays last detected wrong piece
   - Yellow-tinted background
   - Updates when mixed-color pieces are classified

### Controls Panel
- **Start Classification**: Begin video processing
- **Stop Classification**: Stop video processing
- **Reset Counters**: Clear all counts and category displays
- **Real-time Counters**: Live count for each piece type
- **Last Classification Display**: Shows most recent detection with color

## Classification System

### Piece Categories
1. **Proper**: 2 large contours in only one color
2. **Unperforated**: 1 large contour in one color (solid piece)
3. **Wrong**: Mixed colors or incorrect configuration

### Color Detection
The system detects pieces in these colors with specific HSV ranges:
- **Green**: (38-115, 81-255, 14-255)
- **Blue**: (80-128, 58-224, 24-255) 
- **Yellow**: (20-119, 14-255, 38-228)
- **Red**: (0-186, 0-255, 62-255)
- **Purple**: (129-179, 58-255, 24-255)

### Smart Filtering Logic
- Blue pieces may trigger yellow and red detection (false positives)
- System uses area comparison to determine dominant colors
- Exclusion logic removes false positives for accurate classification

## Technical Implementation

### Files Structure
- `main.py`: Application entry point
- `gui.py`: Main GUI interface with 4 video windows
- `camera.py`: Video processing and piece detection
- `classifier.py`: Color detection and piece classification logic
- `logger.py`: Logging system for debugging

### Dependencies
- OpenCV (cv2): Video processing and computer vision
- Tkinter: GUI framework
- PIL (Pillow): Image processing for display
- NumPy: Array operations
- ColorLog: Enhanced logging

## Usage Instructions

### Getting Started
1. Run `python main.py` to start the application
2. Click "Start Classification" to begin processing
3. Monitor the 4 video windows for real-time results
4. Observe counters updating as pieces are detected

### Video Configuration
- Default video: `imgs/video_3.mp4`
- Can be changed in `gui.py` initialization
- Supports various video formats (MP4, AVI, etc.)

### Detection Parameters
- Minimum area threshold: 3000 pixels
- Center detection zone: ±50 pixels from frame center
- Processing frame rate: ~30 FPS
- Color mask blur: 5-pixel median filter

## Troubleshooting

### Common Issues
1. **No video display**: Check video file path in `gui.py`
2. **Poor detection**: Adjust HSV color ranges in `classifier.py`
3. **False positives**: Tune exclusion logic parameters
4. **Performance issues**: Reduce video resolution or processing frequency

### Debug Information
- Console output shows detection details
- Log files stored in `logs/app.log` 
- Real-time classification info displayed in GUI

## Customization

### Adding New Colors
1. Define HSV range in `classifier.py`
2. Add to `color_ranges` dictionary
3. Update exclusion logic if needed

### Adjusting Detection Sensitivity
- Modify area thresholds in `camera.py`
- Adjust contour detection parameters
- Tune center detection zone size

### GUI Layout Changes
- Modify window positions in `gui.py` `createFrames()`
- Adjust video window sizes
- Update button and counter positions

## Future Enhancements
- Multiple video source support
- Export classification results
- Real-time statistics and analytics
- Custom color range configuration UI
- Batch video processing mode
