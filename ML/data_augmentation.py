import os
import cv2
import albumentations as A

# Carpeta con las imágenes
input_folder = r"D:\SumClasses\ArtVi\ML\img"
output_folder = r"D:\SumClasses\ArtVi\ML\img_aug"
os.makedirs(output_folder, exist_ok=True)

# Tipos de dientes (subcarpetas esperadas)
tooth_types = ['inf_izq', 'inf_der', 'cen_izq', 'cen_der', 'ant_sup', 'can']
    
# Ángulos de rotación (0° a 360° en pasos de 5°)
rotation_angles = list(range(0, 360, 5))  # [0, 5, 10, 15, ..., 355]

# Procesar cada tipo de diente
for tooth_type in tooth_types:
    input_type_folder = os.path.join(input_folder, tooth_type)
    
    # Verificar si la carpeta del tipo existe
    if not os.path.exists(input_type_folder):
        print(f"⚠️  Carpeta no encontrada: {input_type_folder}")
        continue
    
    # Crear carpeta de salida para este tipo
    output_type_folder = os.path.join(output_folder, tooth_type)
    os.makedirs(output_type_folder, exist_ok=True)
    
    print(f"📁 Procesando tipo: {tooth_type}")
    
    # Procesar cada imagen en la carpeta del tipo
    image_counter = 0
    for filename in os.listdir(input_type_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_type_folder, filename)
            print(f"  Procesando: {filename}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Error al cargar imagen: {image_path}")
                continue

            # Obtener extensión del archivo
            file_extension = os.path.splitext(filename)[1]
            
            # Copiar imagen original
            original_name = f"{tooth_type}_original_{image_counter}{file_extension}"
            original_output_path = os.path.join(output_type_folder, original_name)
            cv2.imwrite(original_output_path, image)

            # Crear rotaciones
            for i, angle in enumerate(rotation_angles, 1):
                # Crear transformación de rotación específica
                rotate_transform = A.Compose([
                    A.Rotate(limit=[angle, angle], p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0)
                ])
                
                # Aplicar transformación
                augmented = rotate_transform(image=image)
                image_aug = augmented['image']

                # Guardar imagen con nombre numerado
                new_filename = f"{tooth_type}_{image_counter}_{i}{file_extension}"
                new_image_path = os.path.join(output_type_folder, new_filename)
                cv2.imwrite(new_image_path, image_aug)

            image_counter += 1
            print(f"  ✅ Procesado: {filename} -> {len(rotation_angles)} rotaciones")
    
    print(f"✅ Completado tipo {tooth_type}: {image_counter} imágenes procesadas")

print(f"✅ Finalizado el proceso de augmentación. Revisa la carpeta: {output_folder}")