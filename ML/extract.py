import cv2
import numpy as np
from glob import glob
import xlsxwriter

vector_folders_nums =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
row = 0
col = 1

workbook = xlsxwriter.Workbook('caractNums.xlsx')
worksheet = workbook.add_worksheet('caracts')
vector_caracts = np.array([])

def binarizeImg(imgGray):
    return cv2.inRange(imgGray, 0, 127)

def extract_contours(imgBinary):
    cnt = []
    contours, _ = cv2.findContours(imgBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if(len(contours) != 0):
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 100:
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
    
    # roi_1 = imgRoiBin[0:10, 0:10]
    # count_bin = cv2.countNonZero(roi_1)/100
    # x11_roi1 = count_bin
    
    # Divide the image into 4x6 grid of 10x10 ROIs
    roi_features = []
    for i in range(6):  # rows
        for j in range(4):  # columns
            roi = imgRoiBin[i*10:(i+1)*10, j*10:(j+1)*10]
            count_bin = cv2.countNonZero(roi) / 100  # normalize
            roi_features.append(count_bin)

    list_car = [x1_area, x2_perimeter, x3_circularity, x4_hu1, x5_hu2, x6_hu3, x7_hu4, x8_hu5, x9_hu6, x10_hu7] + roi_features
    
    features = np.array(list_car, dtype=np.float32)
    
    return features

def extract_caracts():
    global row, col, vector_caracts
    for n in range(0, len(vector_folders_nums)):
        for img_path in glob('num/' + vector_folders_nums[n] + '/*.png'):
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
                imgRoiBin = cv2.resize(imgRoiBin, (40, 60))   #This attempts to standardize the size of the character images keeping aspect ratio
                vector_caracts = extractPatterns(imgRoiBin, cnt)
                for caract in vector_caracts:
                    worksheet.write(row, 0, n)
                    worksheet.write(row, col, caract)
                    col += 1
                col = 1
                row += 1

            cv2.imshow('imgBinary', cv2.resize(imgBinary, (320, 240)))
            cv2.waitKey(1)
            
    cv2.destroyAllWindows()
    workbook.close()
    print("Excel file created successfully.")
    
if __name__ == "__main__":
    extract_caracts()
    