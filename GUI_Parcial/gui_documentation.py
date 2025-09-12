"""
Documentation for the Updated GUI with 4 Video Windows

LAYOUT DESCRIPTION:
===================

The GUI now has 4 video windows arranged in a 2x2 grid:

1. UPPER LEFT - Main Video Feed
   - Shows the live video feed from the camera/video file
   - Displays detection rectangles and processing in real-time
   - This is where you see the raw video with bounding boxes around detected pieces

2. UPPER RIGHT - Proper Pieces 
   - Shows the last detected PROPER piece (2 holes, correct configuration)
   - Background color: Light Green
   - Updates each time a proper piece is classified

3. LOWER LEFT - Unperforated Pieces
   - Shows the last detected UNPERFORATED piece (solid piece with no holes)
   - Background color: Light Red/Coral
   - Updates each time an unperforated piece is classified

4. LOWER RIGHT - Wrong Pieces
   - Shows the last detected WRONG piece (mixed colors or incorrect configuration)
   - Background color: Light Yellow
   - Updates each time a wrong piece is classified

COUNTERS:
=========
- Unperforated: Count of pieces with 1 large contour in one color
- Wrong: Count of pieces with mixed colors or incorrect configuration  
- Proper: Count of pieces with 2 large contours in only one color
- Last Classification: Shows the most recent classification with color detected

CONTROLS:
=========
- Start Classification: Begin processing the video and classifying pieces
- Stop Classification: Stop the video processing
- Reset Counters: Reset all counters to zero and clear the category video displays

COLORS DETECTED:
===============
The system can detect and classify pieces in these colors:
- Green, Blue, Yellow, Red, Purple

CLASSIFICATION LOGIC:
====================
- Uses HSV color ranges to detect different colored pieces
- Applies exclusion logic to handle overlapping color detections
- Classifies based on contour count and color consistency
- Blue pieces may trigger yellow/red detection, so exclusion logic filters false positives

HOW TO USE:
===========
1. Click "Start Classification" to begin
2. Watch the main video feed (upper left) for live processing
3. Observe the category-specific windows update as pieces are detected
4. Monitor the counters on the right side
5. Use "Reset Counters" to clear results and start fresh
"""

print("GUI Documentation loaded successfully!")
print("\nThe GUI now supports 4 video windows:")
print("- Upper Left: Main video feed with detection rectangles")
print("- Upper Right: Last detected PROPER piece") 
print("- Lower Left: Last detected UNPERFORATED piece")
print("- Lower Right: Last detected WRONG piece")
print("\nEach category window shows the processed frame when that type of piece is detected.")
