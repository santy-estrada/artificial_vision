import os
import shutil

# Ruta de origen con los .jpg y .txt
source_folder = r"D:\SumClasses\ArtVi\DeepLearning\g2\output_labeled_aug"

# Base directory donde crear la estructura train
base_dir = r"D:\SumClasses\ArtVi\DeepLearning\g2"

# Rutas destino (dentro de g2)
dest_images = os.path.join(base_dir, 'test', 'images')
dest_labels = os.path.join(base_dir, 'test', 'labels')

# Crear carpetas destino si no existen
os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_labels, exist_ok=True)

# Listar y ordenar archivos para mantener orden consistente
files = sorted(os.listdir(source_folder))

for file_name in files:
    source_file = os.path.join(source_folder, file_name)

    if file_name.lower().endswith('.jpg'):
        # Copiar imagen con nombre original
        shutil.copy2(source_file, os.path.join(dest_images, file_name))
        print(f"Copiado imagen: {file_name} -> {dest_images}")

    elif file_name.lower().endswith('.txt'):
        # Copiar txt con nombre original
        shutil.copy2(source_file, os.path.join(dest_labels, file_name))
        print(f"Copiado label: {file_name} -> {dest_labels}")
