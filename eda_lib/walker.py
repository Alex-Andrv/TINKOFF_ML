import base64

import cv2
import os

from tqdm import tqdm

from eda_lib.clip_model import ClipModel

from PIL import Image

def get_image_sizes(root_dir):
    image_sizes = []

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)

            # Проверяем, что файл имеет расширение изображения (например, .jpg, .png)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                try:
                    # Считываем изображение с помощью OpenCV
                    img = cv2.imread(file_path)
                    if img is not None:
                        height, width, _ = img.shape
                        image_sizes.append({
                            'file_path': file_path,
                            'width': width,
                            'height': height
                        })
                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {str(e)}")

    return image_sizes

def get_images_tags(root_dir, model: ClipModel):
    image_tags = []

    for foldername, subfolders, filenames in tqdm(os.walk(root_dir)):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            dir_name, file_name = os.path.split(file_path)
            sub_dir, source = os.path.split(dir_name)
            sub_sub_dir, hotel_id = os.path.split(sub_dir)
            _, chain_id = os.path.split(sub_sub_dir)
            # Проверяем, что файл имеет расширение изображения (например, .jpg, .png)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                try:
                    tags = model.get_tags(file_path)

                    image_tags.append({
                        'file_name': file_name,
                        'source': source,
                        'hotel_id': int(hotel_id),
                        'chain_id': int(chain_id),
                        'file_path': file_path,
                        'tags': tags
                    })
                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {str(e)}")

    return image_tags

def get_images(root_dir, file_names):
    images = []

    for foldername, subfolders, filenames in tqdm(os.walk(root_dir)):
        for filename in filenames:
            if filename in file_names:
                file_path = os.path.join(foldername, filename)
                dir_name, file_name = os.path.split(file_path)
                sub_dir, source = os.path.split(dir_name)
                sub_sub_dir, hotel_id = os.path.split(sub_dir)
                _, chain_id = os.path.split(sub_sub_dir)
                # Проверяем, что файл имеет расширение изображения (например, .jpg, .png)
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    try:
                        img = cv2.imread(file_path)
                        height, width, _ = img.shape
                        images.append({
                            'file_name': file_name,
                            'source': source,
                            'hotel_id': int(hotel_id),
                            'chain_id': int(chain_id),
                            'file_path': file_path,
                            'width': width,
                            'height': height
                        })
                    except Exception as e:
                        print(f"Ошибка при обработке файла {file_path}: {str(e)}")

    return images
