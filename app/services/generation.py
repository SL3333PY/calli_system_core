import os
import random
import cv2
import numpy as np

# request
# 1. text
# 2. calligrapher
# response
# 1. generated image


def generate_calligraphy(text, calligrapher):
    directory = r'../calli_data/' + calligrapher
    target = text
    image_list = [file for file in os.listdir(directory) if file.startswith(target)]
    file = str(random.choice(image_list))
    image = cv2.imdecode(np.fromfile(file=os.path.join(directory,file), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return image
