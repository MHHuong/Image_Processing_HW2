import cv2
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import time
def apply_negative(image):
    np_img = np.array(image)
    np_negative = 255 - np_img
    image = Image.fromarray(np_negative)
    return image

def apply_log(image, c, logarit):
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb, dtype=np.float64)
    s = c * np.log(1.0 + img_array)
    s = np.clip(s, 0, 255)
    return Image.fromarray(s.astype(np.uint8), 'RGB')

def apply_gamma(image, c, gamma):
    img = np.array(image, dtype=np.float64) / 255.0
    s = c * np.power(img, gamma)
    s = np.clip(s * 255.0, 0, 255)
    return Image.fromarray(s.astype(np.uint8), 'RGB')

def apply_piecewise_linear(image, r_min, r_max):
    img = np.array(image, dtype=np.float64)
    stretched = (img - r_min) * (255.0 / (r_max - r_min))
    stretched = np.clip(stretched, 0, 255)
    return Image.fromarray(stretched.astype(np.uint8), 'RGB')

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

def apply_avg_filter(image, n):
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

def apply_median_filter(image, n):
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

def apply_gauss_filter(image, n, sigma, return_timing=False):
    import time
    
    t_start = time.time()
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
    
    total_time = time.time() - t_start
    
    if return_timing:
        return imgC, total_time
    return imgC

def apply_max_min_filter(image, n, filter_type='min'):
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

def apply_midpoint_filter(image, n):
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
    
def create_D_matrix(rows, cols):
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - center_row)**2 + (V - center_col)**2)
    return D

def apply_frequency_filter(img_gray, H_filter_func, D0, n=None, return_timing=False):
    import time
    import cv2 as cv
    
    rows, cols = img_gray.shape
    
    # Đo thời gian chuyển sang miền tần số
    t_start_fft = time.time()
    dft = cv.dft(np.float32(img_gray), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    t_fft = time.time() - t_start_fft
    
    # Đo thời gian xử lý filter
    t_start_filter = time.time()
    if n is None:
        H = H_filter_func(rows, cols, D0)
    else:
        H = H_filter_func(rows, cols, D0, n)
    
    H_complex = np.zeros_like(dft_shift)
    H_complex[:,:,0] = H
    H_complex[:,:,1] = H
    
    G_shift = dft_shift * H_complex
    t_filter = time.time() - t_start_filter
    
    # Đo thời gian chuyển về miền không gian
    t_start_ifft = time.time()
    G_ishift = np.fft.ifftshift(G_shift)
    img_back = cv.idft(G_ishift)
    img_back = cv.magnitude(img_back[:,:,0], img_back[:,:,1])
    cv.normalize(img_back, img_back, 0, 255, cv.NORM_MINMAX)
    img_out = np.uint8(img_back)
    t_ifft = time.time() - t_start_ifft
    
    if return_timing:
        timing_info = {
            'fft_time': t_fft,
            'filter_time': t_filter,
            'ifft_time': t_ifft,
            'total_time': t_fft + t_filter + t_ifft
        }
        return img_out, timing_info
    
    return img_out

def apply_IHPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.ones((rows, cols))
    H[D <= D0] = 0
    return H

def apply_ILPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.zeros((rows, cols))
    H[D <= D0] = 1 
    return H

def apply_BLPF(rows, cols, D0, n=2):
    D = create_D_matrix(rows, cols)
    H = 1 / (1 + (D / D0)**(2 * n))
    return H

def apply_BHPF(rows, cols, D0, n=2):
    D = create_D_matrix(rows, cols)
    D[D == 0] = 1e-6
    H = 1 / (1 + (D0 / D)**(2 * n))
    return H

def apply_GLPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.exp(-(D**2) / (2 * (D0**2)))
    return H

def apply_GHPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = 1 - np.exp(-(D**2) / (2 * (D0**2)))
    return H