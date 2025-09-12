import torch
import os

print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Shows bundled CUDA runtime
print(torch.cuda.get_device_name(0))

cat = {0 : "car", 1: "truck"}
print(cat[0])
cat[0] = "cmabio"
print(cat[0])

path = r"D:\SumClasses\ArtVi\DeepLearning\g2\trucks"

# # Ruta de la imagen que deseas predecir
# image_path = r"D:\SumClasses\ArtVi\DeepLearning\g2\trucks\06-01-42_002_4_scale_2.5.jpg"  # Cambia esto por tu imagen

# Realizar la predicción
for file_name in os.listdir(path):
    if file_name.lower().endswith(".jpg"):
        print(path+file_name)
for c,v in cat.items():
    print(f"clave: {c}; valor {v}")

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

print(limits[1][0])