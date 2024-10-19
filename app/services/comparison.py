import cv2
import numpy as np

# 1. 讓尺寸相同(相對位置相同)
# 2. 灰階化、二值化
# 3. 比較


def match_images(image1, image2):
    # image2 to red
    red_color = (0, 0, 255)  # 红色的BGR值
    binary_image2 = convert_to_binary(image2)
    image2_colored = cv2.cvtColor(binary_image2, cv2.COLOR_GRAY2BGR)
    image2_colored[np.where(binary_image2 != 255)] = red_color

    # overlap
    binary_image1 = convert_to_binary(image1)
    image1_colored = cv2.cvtColor(binary_image1, cv2.COLOR_GRAY2BGR)
    overlapped_image = cv2.addWeighted(image2_colored, 0.4, image1_colored, 0.6, 0)
    return overlapped_image


def process_image(image, target_size):
    if image.shape[0] != image.shape[1]:
        image = resize_to_square(image)
    resized_image = cv2.resize(image, (target_size, target_size))
    # 取得邊緣
    region_rect = find_edge(resized_image)

    return resized_image, region_rect


def convert_to_binary(image, contrast=200, brightness=150):

    # 調整對比度和亮度
    contrast_image = image * (contrast / 127 + 1) - contrast + brightness
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)

    # 模糊化處理
    blurred_image = cv2.GaussianBlur(contrast_image, (5, 5), 0)

    # 二值化圖片
    _, binary_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY)

    return binary_image


def find_edge(image):
    binary_image = convert_to_binary(image)

    # 找輪廓需要黑白顛倒的圖
    changed_image = 255 - binary_image

    # 找到圖片的輪廓
    contours, _ = cv2.findContours(changed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 找到所有輪廓的邊界框，並將它們合併成一個大的邊界框
    x_min = min([cv2.boundingRect(contour)[0] for contour in contours])
    y_min = min([cv2.boundingRect(contour)[1] for contour in contours])
    x_max = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in contours])
    y_max = max([cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in contours])

    # 根據邊界框裁剪圖片
    # cropped_image = contrast_image[y_min:y_max, x_min:x_max]

    return [y_min, y_max, x_min, x_max]


def stretch_image_region(image1, image2, region_rect1, region_rect2, target_width, target_height):
    y2_min, y2_max, x2_min, x2_max = region_rect2

    top_diff = y2_min
    bot_diff = 256 - y2_max
    left_diff = x2_min
    right_diff = 256 - x2_max

    # 裁切圖一
    y1_min, y1_max, x1_min, x1_max = region_rect1
    region_image1 = image1[y1_min-top_diff:y1_max+bot_diff, x1_min-left_diff:x1_max+right_diff]

    # 填成方形
    region_image1 = resize_to_square(region_image1)

    # resize
    image_resize = cv2.resize(region_image1, (target_width, target_height))

    return image_resize


def resize_to_square(image):
    size = image.shape
    h = size[0]
    w = size[1]
    if w == h:
        return image
    if w > h:
        pad = (w - h) // 2
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return image
    else:
        pad = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return image
