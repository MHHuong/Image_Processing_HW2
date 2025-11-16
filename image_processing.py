import cv2
from PIL import Image, ImageOps, ImageFilter
import numpy as np

def apply_negative(image):
    np_image = np.array(image)
    np_negative = 255 - np_image
    image = Image.fromarray(np_negative)
    return image

def apply_log(image, c, logarit):
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb, dtype=np.float64)
    s = c * (np.log(1.0 + img_array) / np.log(float(logarit)))
    s = np.clip(s, 0, 255)
    return Image.fromarray(s.astype(np.uint8), 'RGB')

def apply_gamma(image, c, gamma):
    img = np.array(image, dtype=np.float64) / 255.0
    s = c * np.power(img, gamma)
    s = np.clip(s * 255.0, 0, 255)
    return Image.fromarray(s.astype(np.uint8))

def apply_piecewise_linear(image, r_min, r_max):
    img = np.array(image, dtype=np.float64)
    stretched = (img - r_min) * (255.0 / (r_max - r_min))
    stretched = np.clip(stretched, 0, 255)
    return Image.fromarray(stretched.astype(np.uint8))

def conv(A,k,b=0):
    kh, kw = k.shape
    if b>0:
        h, w = A.shape
        B = np.ones((h+kh-1, w+kw-1))
        th = int(kh/2)
        tw = int(kw/2)
        B[th:h+th, tw:w+tw] = A
        A = B
    h, w = A.shape
    C = np.ones((h, w))
    for i in range(0, h - kh + 1):
        for j in range(0, w - kw + 1):
            sA = A[i:i+kh, j:j+kw]
            C[i, j] = np.sum(k*sA)
    C = C[0:h - kh + 1, 0:w - kw + 1]
    return C

def apply_avg_filter(image, n):
    k = np.ones((n, n)) / (n ** 2)
    r, g, b = cv2.split(image)
    B = conv(b, k, 1)
    G = conv(g, k, 1)
    R = conv(r, k, 1)
    imgC = np.array(cv2.merge((R, G, B)), dtype='uint8')
    return imgC

def apply_median_filter(image, n):
    if isinstance(image, np.ndarray):
        img = image.astype(np.float64)
        is_bgr = True
    else:
        img = np.array(image.convert('RGB'), dtype=np.float64)
        is_bgr = False
    
    h, w, c = img.shape
    out = np.zeros((h, w, c))
    k = n // 2  
    for channel in range(c):
        channel_data = img[:, :, channel]
        for i in range(k, h - k):
            for j in range(k, w - k):
                window = channel_data[i-k:i+k+1, j-k:j+k+1]
                out[i, j, channel] = np.median(window)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if is_bgr:
        return out  
    else:
        return Image.fromarray(out, mode='RGB')  

def gauss_kernel (l, sig):
    s = round((l - 1) / 2)
    ax = np.linspace(-s, s, 1)
    gauss = np.exp(-np.square(ax) / (2 * (sig **2)))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def apply_gauss_filter(image, kernel):
    img = np.array(image.convert('RGB'), dtype=np.float64)
    k = gauss_kernel(k_size, sigma)
    r, g, b = cv2.split(img)
    R = conv(r, k, 1)
    G = conv(g, k, 1)
    B = conv(b, k, 1)
    img_blur = cv2.merge((R, G, B))
    img_blur = np.clip(img_blur, 0, 255).astype(np.uint8)
    return Image.fromarray(img_blur, 'RGB')