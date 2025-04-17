import os
import cv2
import numpy as np

def load_and_preprocess_images(input_dir, target_size=(128, 128)):
    images = []
    for file in os.listdir(input_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0
            images.append(img)
    return np.array(images)
