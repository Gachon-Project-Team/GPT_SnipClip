import numpy as np
import cv2


def generate_gaussian_noise(shape, mean=0, std=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(mean, std, shape)
    return noise.astype(np.float32)

def generate_salt_pepper_noise(image_shape, noise_ratio=0.03, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # 노이즈 배열 초기화
    noise = np.zeros(image_shape, dtype=np.uint8)
    total_pixels = np.prod(image_shape[:2])  # H × W
    
    # Salt noise (흰색 픽셀, 255)
    salt_pixels = int(total_pixels * noise_ratio / 2)
    if len(image_shape) == 3:  # 컬러 이미지
        salt_coords_y = np.random.randint(0, image_shape[0], salt_pixels)
        salt_coords_x = np.random.randint(0, image_shape[1], salt_pixels)
        noise[salt_coords_y, salt_coords_x, :] = 255
    else:  # 그레이스케일 이미지
        salt_coords_y = np.random.randint(0, image_shape[0], salt_pixels)
        salt_coords_x = np.random.randint(0, image_shape[1], salt_pixels)
        noise[salt_coords_y, salt_coords_x] = 255
    
    # Pepper noise (검은색 픽셀, 명시적 처리)
    pepper_pixels = int(total_pixels * noise_ratio / 2)
    if len(image_shape) == 3:  # 컬러 이미지
        pepper_coords_y = np.random.randint(0, image_shape[0], pepper_pixels)
        pepper_coords_x = np.random.randint(0, image_shape[1], pepper_pixels)
        noise[pepper_coords_y, pepper_coords_x, :] = 0
    else:  # 그레이스케일 이미지
        pepper_coords_y = np.random.randint(0, image_shape[0], pepper_pixels)
        pepper_coords_x = np.random.randint(0, image_shape[1], pepper_pixels)
        noise[pepper_coords_y, pepper_coords_x] = 0
    
    return noise

def apply_salt_pepper(image, noise_ratio=0.03, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    noisy_image = image.copy()
    total_pixels = image.size
    
    # Salt noise (흰색 픽셀)
    salt_pixels = int(total_pixels * noise_ratio / 2)
    salt_coords = [np.random.randint(0, i, salt_pixels) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255
    
    # Pepper noise (검은색 픽셀)
    pepper_pixels = int(total_pixels * noise_ratio / 2)
    pepper_coords = [np.random.randint(0, i, pepper_pixels) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0
    
    return noisy_image

def generate_blur_effect(image, kernel_size=(5, 5), sigma=1.5):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred

def apply_gaussian_noise(image, mean=0, std=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    noise = generate_gaussian_noise(image.shape, mean, std, seed=None)  # 시드는 이미 설정됨
    noisy_image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    return noisy_image

def apply_median_filter_attack(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def apply_jpeg_compression_simulation(image, quality=50):
    # OpenCV의 JPEG 인코딩/디코딩으로 압축 시뮬레이션
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    return compressed_image

def apply_rotation_attack(image, angle=5):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전 행렬 생성
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 회전 적용
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image

def apply_scaling_attack(image, scale_factor=0.8):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # 축소 후 다시 원본 크기로 확대
    scaled_down = cv2.resize(image, (new_w, new_h))
    scaled_image = cv2.resize(scaled_down, (w, h))
    
    return scaled_image

def apply_histogram_equalization_attack(image):
    if len(image.shape) == 3:
        # 컬러 이미지의 경우 YUV 색공간에서 Y채널만 평활화
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        equalized_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # 그레이스케일 이미지
        equalized_image = cv2.equalizeHist(image)
    
    return equalized_image

def apply_gamma_correction_attack(image, gamma=0.7):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # 룩업 테이블 적용
    gamma_corrected_image = cv2.LUT(image, table)
    
    return gamma_corrected_image

def calculate_nc(image1, image2):
    try:
        img1_flat = image1.flatten().astype(np.float64)
        img2_flat = image2.flatten().astype(np.float64)
        
        correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        return max(0.0, correlation)  # 음수 방지
    except:
        return 0.0

def calculate_psnr(image1, image2):
    mse = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
    if mse < 1e-8:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(image1, image2):
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    
    sigma1_sq = np.var(image1)
    sigma2_sq = np.var(image2)
    sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim

# 노이즈 적용 통합 함수
def apply_noise_attack(image, noise_type='none', seed=None, **kwargs):
    if noise_type == 'none':
        return image.copy()
    
    elif noise_type == 'gaussian':
        std = kwargs.get('std', 3)
        mean = kwargs.get('mean', 0)
        return apply_gaussian_noise(image, mean=mean, std=std, seed=seed)
    
    elif noise_type == 'salt_pepper':
        noise_ratio = kwargs.get('noise_ratio', 0.03)
        return apply_salt_pepper(image, noise_ratio=noise_ratio, seed=seed)
    
    elif noise_type == 'blur':
        kernel_size = kwargs.get('kernel_size', (5, 5))
        sigma = kwargs.get('sigma', 1.5)
        return generate_blur_effect(image, kernel_size=kernel_size, sigma=sigma)
    
    elif noise_type == 'jpeg':
        quality = kwargs.get('quality', 50)
        return apply_jpeg_compression_simulation(image, quality=quality)
    
    elif noise_type == 'rotation':
        angle = kwargs.get('angle', 5)
        return apply_rotation_attack(image, angle=angle)
    
    elif noise_type == 'scaling':
        scale_factor = kwargs.get('scale_factor', 0.8)
        return apply_scaling_attack(image, scale_factor=scale_factor)
    
    elif noise_type == 'histogram':
        return apply_histogram_equalization_attack(image)
    
    elif noise_type == 'gamma':
        gamma = kwargs.get('gamma', 0.7)
        return apply_gamma_correction_attack(image, gamma=gamma)
    
    elif noise_type == 'median':
        kernel_size = kwargs.get('kernel_size', 3)
        return apply_median_filter_attack(image, kernel_size=kernel_size)
    
    else:
        return image.copy()