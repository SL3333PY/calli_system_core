import cv2
from skimage.metrics import structural_similarity
import numpy as np

from app.services.comparison import convert_to_binary


def get_similarity(img1, img2):
    ssim = get_ssim(img1, img2)
    aHash1 = get_aHash(img1)
    aHash2 = get_aHash(img2)
    aHash = compare_hash(aHash1, aHash2)
    dHash1 = get_dHash(img1)
    dHash2 = get_dHash(img2)
    dHash = compare_hash(dHash1, dHash2)
    pHash1 = get_pHash(img1)
    pHash2 = get_pHash(img2)
    pHash = compare_hash(pHash1, pHash2)
    return ssim, aHash, dHash, pHash


def get_ssim(img1, img2):
    img1 = convert_to_binary(img1)
    img2 = convert_to_binary(img2)
    return structural_similarity(img1, img2, data_range=255, multichannel=True)


def get_aHash(img):
    # 均值哈希算法
    # 缩放为8*8
    img = cv2.resize(img, (8, 8))
    if len(img.shape) == 3:  # 圖像有 3 個通道 (BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # 圖像已經是單通道 (灰度或二值)
        gray = img
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def get_dHash(img):
    # 差值哈希算法
    # 缩放8*8
    img = cv2.resize(img, (9, 8))
    if len(img.shape) == 3:  # 圖像有 3 個通道 (BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # 圖像已經是單通道 (灰度或二值)
        gray = img
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def get_pHash(img):     # 要 hamming distance
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    if len(img.shape) == 3:  # 圖像有 3 個通道 (BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # 圖像已經是單通道 (灰度或二值)
        gray = img
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    average = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > average:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# Hash值對比
def compare_hash(hash1, hash2):
    distance = 0
    # hash長度不同則返回-1代表傳參出錯
    if len(hash1) != len(hash2):
        return -1
    # 遍歷判斷
    for i in range(len(hash1)):
        # 不相等則n計數+1，n最終為相似度
        if hash1[i] != hash2[i]:
            distance = distance + 1
    return 1 - distance / 64




# # 讀取兩張圖片
# image1_write = cv2.imread('4f8d_write.png', cv2.IMREAD_GRAYSCALE)
# image2_generate = cv2.imread('4f8d_gen.png', cv2.IMREAD_GRAYSCALE)



