import cv2
import numpy as np

# HSV ranges for each color (H_min, H_max), (S_min, S_max), (V_min, V_max)
green_range = ((38, 115), (81, 255), (14, 255))  # HSV range for green color
blue_range = ((80, 128), (58, 224), (24, 255))   # HSV range for blue color
yellow_range = ((20, 119), (14, 255), (38, 228)) # HSV range for yellow color
red_range = ((0, 186), (0, 255), (62, 255))      # HSV range for red color
purple_range = ((129, 179), (58, 255), (24, 255)) # HSV range for purple color

# Color ranges in order of priority (green, blue, yellow, red, purple)
color_ranges = {
    'green': green_range,
    'blue': blue_range,
    'yellow': yellow_range,
    'red': red_range,
    'purple': purple_range
}

# Counters for piece classification
cont_unperforated = 0  # 1 large contour in one color
cont_wrong = 0         # 2 contours in one color, 1 in another (mixed colors)
cont_proper = 0        # 2 large contours in only one color

def get_color_mask(hsv_frame, color_range):
    """Create a binary mask for a specific color range"""
    h_range, s_range, v_range = color_range
    lower_bound = np.array([h_range[0], s_range[0], v_range[0]])
    upper_bound = np.array([h_range[1], s_range[1], v_range[1]])
    
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    # Apply median blur to reduce noise
    mask = cv2.medianBlur(mask, 5)
    return mask

def count_large_contours(mask, min_area=5000):
    """Count contours with area larger than min_area"""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return len(large_contours), contours

def classify_piece(hsv_frame):
    """
    Classify a piece based on contour counts for each color.
    Uses priority-based detection to handle overlapping color ranges.
    Blue pieces may trigger yellow and red, so we use exclusion logic.
    Returns: classification ('unperforated', 'wrong', 'proper', 'unknown'), color detected
    """
    color_contour_counts = {}
    color_areas = {}
    
    # Check each color in order: green, blue, yellow, red, purple
    for color_name, color_range in color_ranges.items():
        mask = get_color_mask(hsv_frame, color_range)
        outside_contour_count, contours = count_large_contours(mask, min_area=100000)
        
        if outside_contour_count > 0:
            # Calculate total area for this color
            total_area = sum([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 300000])
            color_contour_counts[color_name] = outside_contour_count
            color_areas[color_name] = total_area
    
    # Debug print
    if color_contour_counts:
        print(f"  Raw detections: {color_contour_counts}")
        print(f"  Areas: {color_areas}")
    
    # Apply exclusion logic based on your observation:
    # Blue pieces trigger blue, yellow, and red detection
    # Yellow pieces only show yellow
    # Red pieces only show red
    
    filtered_colors = {}
    
    # If blue is detected along with yellow and red, it's likely a blue piece
    if 'blue' in color_contour_counts and 'yellow' in color_contour_counts and 'red' in color_contour_counts:
        # This is likely a blue piece causing false positives
        filtered_colors['blue'] = color_contour_counts['blue']
        print(f"  Filtered: Blue piece detected (removed yellow and red false positives)")
    
    # If only yellow and red are detected together (without blue), check areas
    elif 'yellow' in color_contour_counts and 'red' in color_contour_counts and 'blue' not in color_contour_counts:
        # Compare areas to determine the dominant color
        if color_areas['yellow'] > color_areas['red'] * 1.2:  # Yellow is significantly larger
            filtered_colors['yellow'] = color_contour_counts['yellow']
            print(f"  Filtered: Yellow dominant (area: {color_areas['yellow']:.0f} vs red: {color_areas['red']:.0f})")
        elif color_areas['red'] > color_areas['yellow'] * 1.2:  # Red is significantly larger
            filtered_colors['red'] = color_contour_counts['red']
            print(f"  Filtered: Red dominant (area: {color_areas['red']:.0f} vs yellow: {color_areas['yellow']:.0f})")
        else:
            # Areas are similar, might be a mixed piece
            filtered_colors = {'yellow': color_contour_counts['yellow'], 'red': color_contour_counts['red']}
            print(f"  Filtered: Mixed yellow/red piece (similar areas)")
    
    # If only one color from yellow/red is detected (clean detection)
    elif 'yellow' in color_contour_counts and 'red' not in color_contour_counts and 'blue' not in color_contour_counts:
        filtered_colors['yellow'] = color_contour_counts['yellow']
        print(f"  Filtered: Pure yellow detection")
    elif 'red' in color_contour_counts and 'yellow' not in color_contour_counts and 'blue' not in color_contour_counts:
        filtered_colors['red'] = color_contour_counts['red']
        print(f"  Filtered: Pure red detection")
    
    # For other colors (green, purple), use them as-is
    for color in ['green', 'purple']:
        if color in color_contour_counts:
            filtered_colors[color] = color_contour_counts[color]
    
    # Analyze the filtered contour counts to classify the piece
    if len(filtered_colors) == 0:
        return 'unknown', 'none'
    
    # If only one color has contours
    if len(filtered_colors) == 1:
        color, count = next(iter(filtered_colors.items()))
        if count == 1:
            return 'unperforated', color
        elif count == 2:
            return 'proper', color
        else:
            # More than 2 contours in one color - could be noise or complex shape
            return 'proper' if count >= 2 else 'unperforated', color
    
    # If multiple colors have contours (mixed piece - wrong)
    else:
        # Find the color with the most contours
        primary_color = max(filtered_colors.items(), key=lambda x: x[1])
        total_contours = sum(filtered_colors.values())
        
        # If we have exactly 3 total contours across colors, it's likely a wrong piece
        if total_contours == 3:
            return 'wrong', primary_color[0]
        # If one color dominates, classify based on that color
        elif primary_color[1] >= 2:
            return 'wrong', primary_color[0]
        else:
            # Multiple colors with equal contours - return the dominant one
            return 'wrong', primary_color[0]

capture = cv2.VideoCapture(r'imgs\video_3.mp4')
testigo = True
processed_pieces = []
frame_count = 0

while capture.isOpened():   
    ret, frame = capture.read()
    if not ret:
        print("Video ended")
        break

    frame = cv2.resize(frame, (1280, 720))

    frame_count += 1
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Try each color range to find the best detection mask
    best_mask = None
    best_area = 0
    
    for color_name, color_range in color_ranges.items():
        temp_mask = get_color_mask(hsv_frame, color_range)
        temp_contours, _ = cv2.findContours(temp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if temp_contours:
            temp_areas = [cv2.contourArea(c) for c in temp_contours]
            max_temp_area = max(temp_areas)
            
            if max_temp_area > best_area:
                best_area = max_temp_area
                best_mask = temp_mask
    
    detection_mask = best_mask if best_mask is not None else np.zeros_like(hsv_frame[:,:,0])
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
            
            # Detect when a piece is at the center of the screen (around pixel 640)
            # Allow some tolerance (±50 pixels) around the center
            if 590 <= center_x <= 690 and testigo and max_area > 3000:
                # Create a region of interest around the detected piece for classification
                roi_x1 = max(0, x - 20)
                roi_y1 = max(0, y - 20)
                roi_x2 = min(frame.shape[1], x + w + 20)
                roi_y2 = min(frame.shape[0], y + h + 20)
                
                roi_hsv = hsv_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Draw bounding rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw center line for reference
                cv2.line(frame, (640, 0), (640, frame.shape[0]), (255, 0, 0), 2)
                # Draw center point of the detected object
                cv2.circle(frame, (center_x, y + h // 2), 5, (0, 0, 255), -1)
                
                # Classify the piece using the ROI
                classification, detected_color = classify_piece(roi_hsv)
                
                # Update counters
                if classification == 'unperforated':
                    cont_unperforated += 1
                elif classification == 'wrong':
                    cont_wrong += 1
                elif classification == 'proper':
                    cont_proper += 1
                
                processed_pieces.append((classification, detected_color))
                
                print(f'Frame {frame_count}: Piece detected: {classification} ({detected_color})')
                print(f'  Area: {max_area:.1f}, Center position: x={center_x}, y={y + h // 2}')
                
                # Show the detection
                cv2.imshow("Contours", frame)
                cv2.imshow("Detection Mask", detection_mask)
                
                
                testigo = False
                    
            # Reset testigo when object moves away from center
            elif center_x < 540 or center_x > 740:
                testigo = True

    # Press 'q' to quit early, or let it run through the whole video
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Interrupted by user")
        break

print("\n=== CLASSIFICATION RESULTS ===")
print(f'Total pieces processed: {len(processed_pieces)}')
print(f'Unperforated pieces: {cont_unperforated}')
print(f'Wrong pieces: {cont_wrong}')
print(f'Proper pieces: {cont_proper}')

print("\nDetailed results:")
for i, (classification, color) in enumerate(processed_pieces, 1):
    print(f'Piece {i}: {classification} - {color}')

capture.release()
cv2.destroyAllWindows()