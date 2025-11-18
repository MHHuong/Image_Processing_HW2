import numpy as np
from PIL import Image, ImageOps, ImageFilter

def conv(A, k, b=0):
    kh, kw = k.shape
    if b > 0:
        h, w = A.shape
        B = np.zeros((h + kh - 1, w + kw - 1)) 
        th = int(kh / 2)
        tw = int(kw / 2)
        B[th:h + th, tw:w + tw] = A
        A = B
    
    h, w = A.shape
    C = np.zeros((h, w)) 
    
    for i in range(0, h - kh + 1):
        for j in range(0, w - kw + 1):
            sA = A[i:i + kh, j:j + kw]
            C[i, j] = np.sum(k * sA)
            
    C = C[0:h - kh + 1, 0:w - kw + 1]
    return C

# Transform

def negative(image):
    np_img = np.array(image)
    np_negative = 255 - np_img
    image = Image.fromarray(np_negative)
    return image

def log_transform(image, c, logarit):
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb, dtype=np.float64)
    s = c * np.log(1.0 + img_array)
    s = np.clip(s, 0, 255)
    return Image.fromarray(s.astype(np.uint8), 'RGB')

def gamma_transform(image, c, gamma):
    img = np.array(image, dtype=np.float64) / 255.0
    s = c * np.power(img, gamma)
    s = np.clip(s * 255.0, 0, 255)
    return Image.fromarray(s.astype(np.uint8), 'RGB')

def piecewise_linear_transform(image, r_min, r_max):
    img = np.array(image, dtype=np.float64)
    stretched = (img - r_min) * (255.0 / (r_max - r_min))
    stretched = np.clip(stretched, 0, 255)
    return Image.fromarray(stretched.astype(np.uint8), 'RGB')

# Filter
def average_filter(image, n):
    k = np.ones((n, n)) / (n ** 2)
    r, g, b = image.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    R = conv(r, k, 1)
    G = conv(g, k, 1)
    B = conv(b, k, 1)
    imgC = Image.merge('RGB', (Image.fromarray(R.astype('uint8')),
                               Image.fromarray(G.astype('uint8')),
                               Image.fromarray(B.astype('uint8'))))
    return imgC

def gaussian_filter(image, n, sigma):
    k = np.zeros((n, n))
    mid = n // 2
    for i in range(n):
        for j in range(n):
            x = i - mid
            y = j - mid
            k[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    k /= np.sum(k)
    
    r, g, b = image.split()
    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    R = conv(r, k, 1)
    G = conv(g, k, 1)
    B = conv(b, k, 1)
    imgC = Image.merge('RGB', (Image.fromarray(R.astype('uint8')),
                               Image.fromarray(G.astype('uint8')),
                               Image.fromarray(B.astype('uint8'))))
    return imgC

def median_filter(image, n):
    if isinstance(image, np.ndarray):
        img = image.astype(np.float64)
        is_bgr = True
    else:
        img = np.array(image.convert('RGB'), dtype=np.float64)
        is_bgr = False
    
    h, w, c = img.shape
    out = np.zeros((h, w, c))
    
    pad = n // 2
    padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = padded_img[i:i + n, j:j + n, ch]
                median_val = np.median(region)
                out[i, j, ch] = median_val
    
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    if is_bgr:
        return out
    else:
        return Image.fromarray(out, 'RGB')
    
def max_min_filter(image, n, filter_type='min'):
    if isinstance(image, np.ndarray):
        img = image.astype(np.uint8)
        is_bgr = True
    else:
        img = np.array(image.convert('RGB'), dtype=np.uint8)
        is_bgr = False
    
    h, w, c = img.shape
    s = n // 2
    padded_img = np.pad(img, ((s, s), (s, s), (0, 0)), mode='edge')
    out = np.zeros((h, w, c), dtype=np.uint8)
    
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = padded_img[i:i + n, j:j + n, ch]
                if filter_type == 'min':
                    out[i, j, ch] = np.min(region)
                elif filter_type == 'max':
                    out[i, j, ch] = np.max(region)
    
    if is_bgr:
        return out
    else:
        return Image.fromarray(out, 'RGB')
    
def midpoint_filter(image, n):
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = np.array(image, dtype=np.uint8)
    
    h, w, c = img.shape
    s = n // 2
    padded_img = np.pad(img, ((s, s), (s, s), (0, 0)), mode='edge')
    Imid = np.zeros((h, w, c), np.uint8)
    
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = padded_img[i:i + n, j:j + n, ch]
                Imin = np.min(region)
                Imax = np.max(region)
                Imid[i, j, ch] = (Imin + Imax) / 2
    
    if isinstance(image, np.ndarray):
        return Imid
    else:
        return Image.fromarray(Imid, mode='RGB')
    
#sharpness enhancement
def laplacian_sharpen(image, kernel_type='4-direction', blur_before=False, ksize=11):
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'), dtype=np.float32)
        is_pil = True
    else:
        img = image.astype(np.float32)
        is_pil = False
    
    if blur_before:
        k_blur = np.ones((ksize, ksize)) / (ksize ** 2)
        img = conv(img, k_blur, 1)
    
    if kernel_type == '4-direction':
        k1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    elif kernel_type == '8-direction':
        k1 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    else:
        k1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    
    L1 = conv(img, k1, 1)
    
    sharpened = img - L1
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if is_pil:
        return Image.fromarray(sharpened, 'L')
    else:
        return sharpened

def sobel_filter(image, threshold=None):
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'), dtype=np.float32)
        is_pil = True
    else:
        img = image.astype(np.float32)
        is_pil = False
    
    ky = np.array([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    kx = np.transpose(ky)
    
    Gx = conv(img, kx, 1)
    Gy = conv(img, ky, 1)
    
    Gm = np.sqrt(Gx**2 + Gy**2)
    
    if Gm.max() > 0:
        Gm = (Gm * 255.0 / Gm.max())
    
    if threshold is not None:
        Gm = (Gm > threshold).astype(np.float32) * 255.0
    
    Gm = np.clip(Gm, 0, 255).astype(np.uint8)
    
    if is_pil:
        return Image.fromarray(Gm, 'L')
    else:
        return Gm

# Frequency 

def create_D_matrix(rows, cols):
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - center_row)**2 + (V - center_col)**2)
    return D

def apply_frequency_filter(img_gray, H_filter_func, D0, n=None):
    rows, cols = img_gray.shape
    
    import cv2 as cv
    dft = cv.dft(np.float32(img_gray), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    if n is None:
        H = H_filter_func(rows, cols, D0)
    else:
        H = H_filter_func(rows, cols, D0, n)
    
    H_complex = np.zeros_like(dft_shift)
    H_complex[:,:,0] = H
    H_complex[:,:,1] = H
    
    G_shift = dft_shift * H_complex
    G_ishift = np.fft.ifftshift(G_shift)
    img_back = cv.idft(G_ishift)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])
    cv.normalize(img_back, img_back, 0, 255, cv.NORM_MINMAX)
    img_out = np.uint8(img_back)
    
    return img_out

def IHPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.ones((rows, cols))
    H[D <= D0] = 0
    return H

def ILPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.zeros((rows, cols))
    H[D <= D0] = 1 
    return H

def BLPF(rows, cols, D0, n=2):
    D = create_D_matrix(rows, cols)
    H = 1 / (1 + (D / D0)**(2 * n))
    return H

def BHPF(rows, cols, D0, n=2):
    D = create_D_matrix(rows, cols)
    D[D == 0] = 1e-6
    H = 1 / (1 + (D0 / D)**(2 * n))
    return H

def GLPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.exp(-(D**2) / (2 * (D0**2)))
    return H

def GHPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = 1 - np.exp(-(D**2) / (2 * (D0**2)))
    return H


