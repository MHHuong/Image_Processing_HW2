import numpy as np
import cv2

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

def laplace(img, sharpen=False):
    k4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    def _proc_channel(ch, kernel, b=1, sharpen_flag=False):
        chf = ch.astype(np.float32)
        out = conv(chf, kernel, b)
        
        if sharpen_flag:
            s = chf - out 
            s = np.clip(s, 0, 255)
            return np.array(s, dtype=np.uint8)

        mag = np.abs(out)
        if mag.max() > 0:
            mag = (mag * 255.0 / mag.max())
        mag_u8 = np.array(mag, dtype=np.uint8)
        return mag_u8

    kernel = k4
    b = 1

    if img is None:
        return None

    if img.ndim == 2:
        return _proc_channel(img, kernel, b, sharpen)
    
    bch, gch, rch = cv2.split(img)
    B = _proc_channel(bch, kernel, b, sharpen)
    G = _proc_channel(gch, kernel, b, sharpen)
    R = _proc_channel(rch, kernel, b, sharpen)
    return cv2.merge((B, G, R))

def sobel(img):
    img_f = img.astype(np.float32) 
    
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.flip(kx.T, axis=0)
    
    imGx = conv(img_f, kx, b=1)
    imGy = conv(img_f, ky, b=1)
    
    gradM = np.sqrt(imGx**2 + imGy**2)
    if gradM.max() > 0:
        gradM = gradM * 255.0 / gradM.max()
        
    gradM = np.array(gradM, dtype=np.uint8)
    return gradM

def run_image_enhancement(image_path):
    img_a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_a is None:
        print(f"Lỗi: không thể đọc ảnh từ {image_path}")
        return

    img_b = laplace(img_a, sharpen=False)

    img_c = laplace(img_a, sharpen=True)

    img_d = sobel(img_a)

    k_avg = np.ones((5, 5)) / (5 ** 2)
    img_d_float = img_d.astype(np.float32)
    img_e_float = conv(img_d_float, k_avg, 1)
    img_e = np.clip(img_e_float, 0, 255).astype(np.uint8)

    img_b_norm = img_b.astype(np.float32) / 255.0
    img_e_norm = img_e.astype(np.float32) / 255.0
    img_f_norm = img_b_norm * img_e_norm
    img_f = (img_f_norm * 255.0).astype(np.uint8)

    img_g = cv2.add(img_a, img_f)

    gamma = 2.5
    c = 1.0
    img_g_norm = img_g.astype(np.float32) / 255.0
    img_h_norm = c * (img_g_norm ** gamma)
    img_h_norm = np.clip(img_h_norm, 0, 1.0)
    img_h = (img_h_norm * 255.0).astype(np.uint8)
    images = [img_a, img_b, img_c, img_d, img_e, img_f, img_g, img_h]
    labels = [
        "(a) Original", "(b) Laplacian Mag", "(c) Sharpened (a-b)",
        "(d) Sobel Mag", "(e) Smoothed Sobel", "(f) Mask (b*e)",
        "(g) Sharpened (a+f)", "(h) Final (Power Law)"
    ]
    
    labeled_images = []
    
    for img, label in zip(images, labels):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w, _ = img_bgr.shape
        img_with_label_space = cv2.copyMakeBorder(img_bgr, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv2.putText(img_with_label_space, label, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        labeled_images.append(img_with_label_space)

    row1 = np.hstack((labeled_images[0], labeled_images[1], labeled_images[2], labeled_images[3]))
    row2 = np.hstack((labeled_images[4], labeled_images[5], labeled_images[6], labeled_images[7]))
    
    final_grid = np.vstack((row1, row2))

    cv2.imshow("Image Enhancement Pipeline (a-h) - Horizontal", final_grid) 
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

run_image_enhancement('Anh.jpg') 