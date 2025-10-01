import cv2
import numpy as np
from glob import glob
import xlsxwriter

vector_folders_teeth =['inf_izq', 'inf_der', 'cen_izq', 'cen_der', 'ant_sup', 'can']
row = 0
col = 1

workbook = xlsxwriter.Workbook('caractTeeth.xlsx')
worksheet = workbook.add_worksheet('caracts')
vector_caracts = np.array([])

def binarizeImg(imgGray):
    return cv2.inRange(imgGray, 30, 200)

def extract_contours(imgBinary):
    cnt = []
    contours, _ = cv2.findContours(imgBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if(len(contours) != 0):
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 500:
            return cnt
        else:
            return []
    return cnt

def extractPatterns(imgRoiBin, cnt):
    x1_area = cv2.contourArea(cnt)
    x2_perimeter = cv2.arcLength(cnt, True)
    x3_circularity = (x1_area) / (x2_perimeter * x2_perimeter)
    M = cv2.moments(cnt)
    Hu = cv2.HuMoments(M)
    x4_hu1 = Hu[0][0]
    x5_hu2 = Hu[1][0]
    x6_hu3 = Hu[2][0]
    x7_hu4 = Hu[3][0]
    x8_hu5 = Hu[4][0]
    x9_hu6 = Hu[5][0]
    x10_hu7 = Hu[6][0]
    
    
    # Divide the image into grid for ROI analysis (adjust grid size for new dimensions)
    roi_features = []
    height, width = imgRoiBin.shape
    grid_rows = 8  # Increased from 6 for 80 pixel height
    grid_cols = 6  # Increased from 4 for 60 pixel width
    
    row_step = height // grid_rows
    col_step = width // grid_cols
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            start_row = i * row_step
            end_row = min((i + 1) * row_step, height)
            start_col = j * col_step
            end_col = min((j + 1) * col_step, width)
            
            roi = imgRoiBin[start_row:end_row, start_col:end_col]
            roi_area = roi.shape[0] * roi.shape[1]
            if roi_area > 0:
                count_bin = cv2.countNonZero(roi) / roi_area  # normalize by actual ROI area
                roi_features.append(count_bin)
            else:
                roi_features.append(0.0)

    list_car = [x1_area, x2_perimeter, x3_circularity, x4_hu1, x5_hu2, x6_hu3, x7_hu4, x8_hu5, x9_hu6, x10_hu7] + roi_features
    
    features = np.array(list_car, dtype=np.float32)
    
    return features

def extract_caracts():
    global row, col, vector_caracts
    for n in range(0, len(vector_folders_teeth)):
        for img_path in glob('img_aug/' + vector_folders_teeth[n] + '/*.jpg'):
            imgGray = cv2.imread(img_path, 0)
            imgColor = cv2.imread(img_path, 1)
            
            imgBinary = binarizeImg(imgGray)

            cnt = extract_contours(imgBinary)
            
            if len(cnt)  > 0:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(imgColor, (x,y), (x+w,y+h), (255,0,0), 1)
                cv2.imshow('imgColor', cv2.resize(imgColor, (320, 240)))
                cv2.waitKey(1)
                imgRoiBin = imgBinary[y:y+h, x:x+w]
                imgRoiBin = cv2.copyMakeBorder(imgRoiBin, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                
                # Standardize size for consistent feature extraction (especially important for rotated images)
                # Use larger size than numbers since teeth are more complex shapes
                imgRoiBin = cv2.resize(imgRoiBin, (60, 80))  # Width x Height - larger than 40x60 for teeth
                
                cv2.imshow('imgRoiBin', imgRoiBin)
                cv2.waitKey(1)
                vector_caracts = extractPatterns(imgRoiBin, cnt)
                for caract in vector_caracts:
                    worksheet.write(row, 0, n)
                    worksheet.write(row, col, caract)
                    col += 1
                col = 1
                row += 1

            cv2.imshow('imgBinary', imgBinary)
            cv2.waitKey(1)
            
    cv2.destroyAllWindows()
    workbook.close()
    print("Excel file created successfully.")
    
if __name__ == "__main__":
    extract_caracts()
    