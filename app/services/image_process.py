import base64
import cv2
import numpy as np
import os
import uuid


def convert_image_to_base64(image_np):
    image = cv2.imencode('.png', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code


def convert_base64_to_image(base64_code):
    if base64_code.startswith("data"):
        base64_code = base64_code.split(",")[1]
    image_data = base64.b64decode(base64_code)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    return image


def save_image(image):
    image = cv2.resize(image, (256, 256))
    directory = r'../storage'
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)
    filename = uuid.uuid4().hex + ".png"
    cv2.imwrite(filename, image)
    return directory + "\\" + filename
