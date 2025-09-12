import cv2
import numpy as np

def read_image(image_path):
    """
    Reads an image from the specified path and returns it.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        numpy.ndarray: The image read from the file.
    """
    imageBgr = cv2.imread(image_path, 1)      #flag = 1 (default), bgr; flag = 0, grayscale
    imageGray = cv2.imread(image_path, 0)          #flag = 0, grayscale
    
    return imageBgr, imageGray

def show_image(image, title="Image", wait_time=0, type=0):
    """
    Displays an image in a window.
    
    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the window.
    """
    cv2.imshow(title, image)
    if type == 0:
        cv2.waitKey(wait_time)          # Wait for a key press if 0 or wait for specified milliseconds
        cv2.destroyAllWindows()
    
def resize_image(image):
    """
    Resizes an image.
    
    Args:
        image (numpy.ndarray): The image to resize.
        
    Returns:
        numpy.ndarray: The resized image.
    """
    resized_image = cv2.resize(image, (640, 480))
    return resized_image

def get_image_size(image):
    """
    Returns the size of the image.
    
    Args:
        image (numpy.ndarray): The image whose size is to be returned.
        
    Returns:
        tuple: The dimensions of the image (height, width).
    """
    return image.shape[:2]  # Returns (height, width) for BGR or Grayscale images

def get_pixel_value(image, x, y):
    """
    Returns the pixel value at the specified coordinates.
    
    Args:
        image (numpy.ndarray): The image from which to get the pixel value.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        
    Returns:
        tuple: The pixel value at (x, y) for BGR images or a single value for Grayscale images.
    """
    return image[y, x]  # Note: OpenCV uses (y, x) indexing

def modify_pixel_value(image, x, y, value):
    """
    Modifies the pixel value at the specified coordinates.
    
    Args:
        image (numpy.ndarray): The image in which to modify the pixel value.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        value (tuple or int): The new pixel value (BGR tuple for color images, single int for grayscale).
    """
    image[y, x] = value  # Note: OpenCV uses (y, x) indexing
    
def gray_to_black_and_white(image):
    """
    Converts a grayscale image to black and white.
    
    Args:
        image (numpy.ndarray): The grayscale image to convert.
        
    Returns:
        numpy.ndarray: The black and white image.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image is not a grayscale image.")
    
    img_copy = image.copy()
    gray_size = get_image_size(img_copy)
    for i in range(gray_size[0]):
        for j in range(gray_size[1]):
            if get_pixel_value(img_copy, j, i) < 128:
                modify_pixel_value(img_copy, j, i, 0)
            else:
                modify_pixel_value(img_copy, j, i, 255)
                
    return img_copy

def invert_B_R(image):
    """
    Inverts the Blue and Red channels of a BGR image.
    
    Args:
        image (numpy.ndarray): The BGR image to modify.
        
    Returns:
        numpy.ndarray: The modified image with inverted Blue and Red channels.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image is not a BGR image.")
    
    img_copy = image.copy()
    size = get_image_size(img_copy)
    
    for i in range(size[0]):
        for j in range(size[1]):
            b,g,r = get_pixel_value(img_copy, j, i)
            modify_pixel_value(img_copy, j, i, (r, g, b))
                
    return img_copy

def turn_color(image, color):
    """
    Only keeps the specified color channel in a BGR image.
    
    Args:
        image (numpy.ndarray): The BGR image to modify.
        
    Returns:
        numpy.ndarray: The modified image with inverted Blue and Red channels.
    """
    if len(image.shape) != 3:
        raise ValueError("Input image is not a BGR image.")
    
    img_copy = image.copy()
    size = get_image_size(img_copy)
    
    for i in range(size[0]):
        for j in range(size[1]):
            b,g,r = get_pixel_value(img_copy, j, i)
            if color == 'blue':
                modify_pixel_value(img_copy, j, i, (b, 0, 0))
            elif color == 'green':
                modify_pixel_value(img_copy, j, i, (0, g, 0))
            elif color == 'red':
                modify_pixel_value(img_copy, j, i, (0, 0, r))
                
    return img_copy

def invert_y(image):
    """
    Inverts image with respect to Y

    Args:
        image (numpy.ndarray): The BGR image to modify.

    Returns:
        numpy.ndarray: The modified inverted image.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)
    newImg = np.empty_like(img_copy)

    for i in range(size[0]):
        for j in range(size[1]):
            newImg[i,j] = get_pixel_value(img_copy, size[0] - j - 1, i)
            
    return newImg

def invert_x(image):
    """
    Inverts image with respect to X

    Args:
        image (numpy.ndarray): The BGR image to modify.

    Returns:
        numpy.ndarray: The modified inverted image.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)
    newImg = np.empty_like(img_copy)

    for i in range(size[0]):
        for j in range(size[1]):
            newImg[i,j] = get_pixel_value(img_copy, j, size[1] - 1 - i)
            
    return newImg

def cuadricula(image, n = 0):
    """
    Divides the image into 4: original, blue, green, and red.
    
    Args:
        image (numpy.ndarray): The image on which to draw the grid.
        
    Returns:
        numpy.ndarray: The image with the grid drawn on it.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)
    
    if n <= 0:
        Ydis = size[0] / 2 if size[0] % 2 == 0 else (size[0] // 2) + 1
        Xdis = size[1] / 2 if size[1] % 2 == 0 else (size[1] // 2) + 1
        for i in range(size[0]):
            for j in range(size[1]):
                b,g,r = get_pixel_value(img_copy, j, i)
                if i < Ydis and j >= Xdis:
                    modify_pixel_value(img_copy, j, i, (b, 0, 0))
                elif i >= Ydis and j < Xdis:
                    modify_pixel_value(img_copy, j, i, (0, g, 0))
                elif i >= Ydis and j >= Xdis:
                    modify_pixel_value(img_copy, j, i, (0, 0, r))
    else:       
        
        # n = 4, size = 8x8
        # For i in range(8): (rows)
        #   For j in range(8): (cols)
        #       block_x = j // 4  # 0 for j=0..3, 1 for j=4..7
        #       block_y = i // 4  # 0 for i=0..3, 1 for i=4..7
        #       c = (block_x + block_y) % 3

        # Block positions:
        # (block_y, block_x): (0,0), (0,1), (1,0), (1,1)
        # c values: (0+0)%3=0 (blue), (0+1)%3=1 (green), (1+0)%3=1 (green), (1+1)%3=2 (red)

        # So:
        # Top-left 4x4: blue
        # Top-right 4x4: green
        # Bottom-left 4x4: green
        # Bottom-right 4x4: red

        # For each pixel in each block, only the corresponding color channel is kept.

        for i in range(size[0]): 
            for j in range(size[1]):  
                b, g, r = get_pixel_value(img_copy, j, i)

                # Determine current block position
                block_x = j // n
                block_y = i // n

                # Choose which color to keep
                c = (block_x + block_y) % 3

                if c == 0:
                    modify_pixel_value(img_copy, j, i, (b, 0, 0))  # Blue block
                elif c == 1:
                    modify_pixel_value(img_copy, j, i, (0, g, 0))  # Green block
                elif c == 2:
                    modify_pixel_value(img_copy, j, i, (0, 0, r))  # Red block
                    
                

    return img_copy

def patternX(image, n = 0):
    """
    Divides the image into 4: original, blue, green, and red.
    
    Args:
        image (numpy.ndarray): The image on which to draw the grid.
        
    Returns:
        numpy.ndarray: The image with the grid drawn on it.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)

    for i in range(size[0]): 
        for j in range(size[1]):  
            b, g, r = get_pixel_value(img_copy, j, i)

            # Determine current block position
            block_x = j // n

            # Choose which color to keep
            c = (block_x) % 3

            if c == 0:
                modify_pixel_value(img_copy, j, i, (b, 0, 0))  # Blue block
            elif c == 1:
                modify_pixel_value(img_copy, j, i, (0, g, 0))  # Green block
            elif c == 2:
                modify_pixel_value(img_copy, j, i, (0, 0, r))  # Red block
                
            

    return img_copy
    
def patternY(image, n = 0):
    """
    Divides the image into 4: original, blue, green, and red.
    
    Args:
        image (numpy.ndarray): The image on which to draw the grid.
        
    Returns:
        numpy.ndarray: The image with the grid drawn on it.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)

    for i in range(size[0]): 
        for j in range(size[1]):  
            b, g, r = get_pixel_value(img_copy, j, i)

            # Determine current block position
            block_y = i // n

            # Choose which color to keep
            c = (block_y) % 3

            if c == 0:
                modify_pixel_value(img_copy, j, i, (b, 0, 0))  # Blue block
            elif c == 1:
                modify_pixel_value(img_copy, j, i, (0, g, 0))  # Green block
            elif c == 2:
                modify_pixel_value(img_copy, j, i, (0, 0, r))  # Red block
                
            

    return img_copy

def binary(image, method = 0, rgbMin = (0,0,0), rgbMax = (180,255,255), Umin = 0, Umax = 255):
    if method == 0:
        # Simple binary thresholding
        ret, imgBinary = cv2.threshold(image, Umin, 255, cv2.THRESH_BINARY)
        print("Thresholding complete. Return value:", ret)
    elif method == 1:
        # Grayscale thresholding using cv2.inRange
        imgBinary = cv2.inRange(image, Umin, Umax)
    # The second argument is the threshold value, the third is the maximum value to use with the THRESH_BINARY thresholding type.
    elif method == 2:
        # RGB/HSV thresholding using cv2.inRange
        imgBinary = cv2.inRange(image, rgbMin, rgbMax)

    return imgBinary  # Return the thresholded image, which is the second element of the tuple returned by cv2.threshold

def replace_color(image, old_color_min, old_color_max, new_color, color_space="BGR"):
    """
    Replaces a specific color range in the image with a new color.
    
    Args:
        image (numpy.ndarray): The image in which to replace the color.
        old_color_min (tuple): The minimum color values to detect (e.g., (0, 0, 100) for red in BGR).
        old_color_max (tuple): The maximum color values to detect (e.g., (50, 50, 255) for red in BGR).
        new_color (tuple): The new color to use (B, G, R).
        color_space (str): Color space to work in ("BGR" or "HSV").
        
    Returns:
        numpy.ndarray: The modified image with the color replaced.
    """
    img_copy = image.copy()
    size = get_image_size(img_copy)
    
    if color_space.upper() == "HSV":
        imageHSV = transformSpaceBGR2HSV(img_copy)
        mask = binary(imageHSV, method=2, rgbMin=old_color_min, rgbMax=old_color_max)
        for i in range(size[0]):
            for j in range(size[1]):
                if mask[i, j] > 0:
                    modify_pixel_value(imageHSV, j, i, new_color)
    elif color_space.upper() == "BGR":
        mask = binary(img_copy, method=2, rgbMin=old_color_min, rgbMax=old_color_max)
        for i in range(size[0]):
            for j in range(size[1]):
                if mask[i, j] > 0:
                    modify_pixel_value(img_copy, j, i, new_color)
                    
    elif color_space.upper() == "RGB":
        img_copy = transformSpaceBGR2RGB(img_copy)
        mask = binary(img_copy, method=2, rgbMin=old_color_min, rgbMax=old_color_max)
        for i in range(size[0]):
            for j in range(size[1]):
                if mask[i, j] > 0:
                    modify_pixel_value(img_copy, j, i, new_color)


    return transformSpaceHSV2BGR(imageHSV) if color_space.upper() == "HSV" else img_copy

def create_trackbar(window, type = 0, name = "Trackbar", max_value = 100):
    """
    Creates a trackbar for adjusting the threshold value.
    """
    cv2.namedWindow(window)
    if type == 0:
        # Grayscale thresholding
        cv2.createTrackbar("Umin", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Umax", window, 0, 255, lambda x: None)
        
    elif type == 1:
        # RGB thresholding trackbars
        cv2.createTrackbar("Rmin", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Rmax", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Gmin", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Gmax", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Bmin", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Bmax", window, 0, 255, lambda x: None)
    elif type == 2:
        # HSV thresholding trackbars
        cv2.createTrackbar("Hmin", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Hmax", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Smin", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Smax", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Vmin", window, 0, 255, lambda x: None)
        cv2.createTrackbar("Vmax", window, 0, 255, lambda x: None)
    elif type == 3:
        # Custom trackbars for other purposes
        cv2.createTrackbar(name, window, 0, max_value, lambda x: None)
        
def get_trackbar_values(window, type = 0, name = "Trackbar"):
    """
    Retrieves the current values of the trackbars.
    
    Returns:
        tuple: A tuple containing the values of the trackbars.
    """
    if type == 0:
        # Grayscale thresholding
        Umin = cv2.getTrackbarPos("Umin", window)
        Umax = cv2.getTrackbarPos("Umax", window)
        return (Umin, Umax)
    elif type == 1:
        # BGR thresholding
        Rmin = cv2.getTrackbarPos("Rmin", window)
        Rmax = cv2.getTrackbarPos("Rmax", window)
        Gmin = cv2.getTrackbarPos("Gmin", window)
        Gmax = cv2.getTrackbarPos("Gmax", window)
        Bmin = cv2.getTrackbarPos("Bmin", window)
        Bmax = cv2.getTrackbarPos("Bmax", window)
        return (Bmin, Gmin, Rmin, Bmax, Gmax, Rmax)
    elif type == 2:
        # HSV thresholding
        Hmin = cv2.getTrackbarPos("Hmin", window)
        Hmax = cv2.getTrackbarPos("Hmax", window)
        Smin = cv2.getTrackbarPos("Smin", window)
        Smax = cv2.getTrackbarPos("Smax", window)
        Vmin = cv2.getTrackbarPos("Vmin", window)
        Vmax = cv2.getTrackbarPos("Vmax", window)
        return (Hmin, Smin, Vmin, Hmax, Smax, Vmax)
    elif type == 3:
        # Custom trackbar
        value = cv2.getTrackbarPos(name, window)
        return (value,)
    

def transformSpaceBGR2HSV(image):
    """
    Convert the image to a different color space.
    """
    return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

def transformSpaceHSV2BGR(image):
    """
    Convert the image to a different color space.
    """
    return cv2.cvtColor(image.copy(), cv2.COLOR_HSV2BGR)

def transformSpaceBGR2RGB(image):
    """
    Convert the image to a different color space.
    """
    return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

def drawRectangle(image, x, y, width, height, color=(0, 255, 0), thickness=2, filled = False):
    """
    Draw a rectangle on the image.
    """
    if filled:
        cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)  # -1 for filled rectangle
    else:
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)