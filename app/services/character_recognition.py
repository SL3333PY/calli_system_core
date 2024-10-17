import easyocr


def detect_characters(image):
    image = str(image)
    reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)
    result = reader.readtext(image)
    return result[0][1]
