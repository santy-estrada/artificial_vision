import functions as fn
import cv2
import os


def main():
    # image_path = r"D:\SumClasses\ArtVi\Clase2\imgs\images\nivel_1\AC10_096.bmp"  # Replace with your image path
    min5 = (0, 190, 140)
    max5 = (190, 250, 255)
    min4 = (0, 200, 210)
    max4 = (100, 255, 255)
    min3 = (135, 250, 130)
    max3 = (165, 255, 170)
    min2 = (125, 250, 110)
    max2 = (145, 255, 140)
    min1 = (95, 230, 95)
    max1 = (135, 255, 135)
    
    limits = ((min5, max5), (min4, max4), (min3, max3), (min2, max2), (min1, max1))
    
    path = r"D:\SumClasses\ArtVi\Clase2\imgs\eval_motors\eval_motors"
    
    
    
    conteo = {"N1" : 0, "N2" : 0, "N3": 0, "N4": 0, "N5": 0}

    for file_name in os.listdir(path):
        image_path = os.path.join(path, file_name)
        print(file_name)
        imageBgr, imageGray = fn.read_image(image_path)
        imageHSV = fn.transformSpaceBGR2HSV(imageBgr)
        fn.show_image(imageHSV, title="Image HSV")
        for k in range(5):
            binPic = fn.binary(imageHSV, method=2, rgbMin=limits[k][0], rgbMax=limits[k][1])
            fn.show_image(binPic, title="Binary Image", type=1) 
            
            size = fn.get_image_size(binPic)
            cont = 0
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            for i in range(size[0]):
                for j in range(size[1]):
                    if binPic[i, j] > 0:
                        cont += 1
                        
            # print(cont)
            # print(k)
            # print(limits[k][0])
            # print(limits[k][1])
                        
            if k == 0 and cont >= 14550:
                conteo["N5"] += 1
                print("N5")
                break
                
            elif k == 1 and cont >= 9000:
                conteo["N4"] += 1
                print("N4")
                break
                
            elif k == 2 and cont >= 13300:
                conteo["N3"] += 1
                print("N3")
                break
                
            elif k == 3 and cont >= 9500:
                # binPicAux = fn.binary(imageHSV, method=2, rgbMin=limits[k+1][0], rgbMax=limits[k+1][1])
                # sizeAux = fn.get_image_size(binPic)
                # contAux = 0
            
                # for iaux in range(sizeAux[0]):
                #     for jaux in range(sizeAux[1]):
                #         if binPicAux[iaux, jaux] > 0:
                #             contAux += 1
                            
                # if contAux >= 13450:
                #     conteo["N1"] += 1
                # else:                           
                #     conteo["N2"] += 1
                conteo["N2"] += 1
                print("N2")
                break
                
            elif k == 4 and cont >= 13450:
                conteo["N1"] += 1
                print("N1")
                break
                
            else:
                print("Not classified yet")
                            
                
    for c,v in conteo.items():
        print(str(c) + ": " + str(v))
        print("--------------")
                
                    

        
    # while True:
    #     # vals = fn.get_trackbar_values("Binary Image", type=2)
    #     # lowLim = vals[:3]
    #     # upLim = vals[3:]
    #     # binPic = fn.binary(imageHSV, method=2, rgbMin=lowLim, rgbMax=upLim)
        
    #     binPic = fn.binary(imageHSV, method=2, rgbMin=(125, 250, 110), rgbMax=(145, 255, 140))

    #     fn.show_image(binPic, title="Binary Image", type=1)
    #     size = fn.get_image_size(binPic)
    #     cont = 0
        
    #     for i in range(size[0]):
    #         for j in range(size[1]):
    #             if binPic[i, j] > 0:
    #                 cont += 1
                    
        
    #     # print(cont)
    
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    
   

if __name__ == "__main__":
    # fn.create_trackbar("Binary Image", type=2)
    # cv2.waitKey(10)  # Ensure the trackbar is created before proceeding
    
    main()
    print("Program completed successfully.")