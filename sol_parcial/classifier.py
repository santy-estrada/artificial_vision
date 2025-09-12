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
    
    # Apply exclusion logic based on observation:
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

def get_best_detection_mask(hsv_frame):
    """
    Find the best detection mask by trying each color range.
    Returns the mask with the largest contour area.
    """
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
    
    return best_mask if best_mask is not None else np.zeros_like(hsv_frame[:,:,0])
