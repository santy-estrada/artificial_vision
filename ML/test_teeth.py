import joblib
import cv2
import numpy as np
from glob import glob
import os

# Class names corresponding to indices
class_names = ['inf_izq', 'inf_der', 'cen_izq', 'cen_der', 'ant_sup', 'can']

def binarizeImg(imgGray):
    return cv2.inRange(imgGray, 10, 110)

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

def extract_features_from_image(img_path):
    """Extract features from a single image"""
    imgGray = cv2.imread(img_path, 0)
    if imgGray is None:
        print(f"Error loading image: {img_path}")
        return None
        
    imgBinary = binarizeImg(imgGray)
    cnt = extract_contours(imgBinary)
    
    if len(cnt) > 0:
        x, y, w, h = cv2.boundingRect(cnt)
        imgRoiBin = imgBinary[y:y+h, x:x+w]
        imgRoiBin = cv2.copyMakeBorder(imgRoiBin, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        features = extractPatterns(imgRoiBin, cnt)
        return features
    else:
        print(f"No valid contour found in: {img_path}")
        return None

# Load models
print("Loading models...")
model_mlp = joblib.load('teeth_classification_mlp.pkl')
model_svm = joblib.load('teeth_classification_svm.pkl')

# Test images folder
test_folder = 'img_test'

# Check if test folder exists
if not os.path.exists(test_folder):
    print(f"Test folder '{test_folder}' not found!")
    exit()

print("Processing test images...")
print("=" * 60)

# Process all images in test folder
for img_path in glob(os.path.join(test_folder, '*')):
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"\nTesting image: {os.path.basename(img_path)}")
        
        # Extract features
        features = extract_features_from_image(img_path)
        
        if features is not None:
            # Reshape for prediction (models expect 2D array)
            features = features.reshape(1, -1)
            
            # Make predictions
            pred_mlp = model_mlp.predict(features)[0]
            pred_svm = model_svm.predict(features)[0]
            
            # Get prediction probabilities (if available)
            try:
                # For MLP, get probability scores
                prob_mlp = model_mlp.predict_proba(features)[0]
                confidence_mlp = max(prob_mlp)
            except:
                confidence_mlp = "N/A"
            
            try:
                # For SVM, get decision function scores
                decision_svm = model_svm.decision_function(features)[0]
                confidence_svm = max(decision_svm) if isinstance(decision_svm, np.ndarray) else decision_svm
            except:
                confidence_svm = "N/A"
            
            print(f"  MLP Prediction: {class_names[pred_mlp]} (confidence: {confidence_mlp})")
            print(f"  SVM Prediction: {class_names[pred_svm]} (confidence: {confidence_svm})")
            
            # Show agreement
            if pred_mlp == pred_svm:
                print(f"  ✅ Both models agree: {class_names[pred_mlp]}")
            else:
                print(f"  ⚠️  Models disagree - MLP: {class_names[pred_mlp]}, SVM: {class_names[pred_svm]}")
        else:
            print("  ❌ Could not extract features from this image")

print("\n" + "=" * 60)
print("Testing completed!")

