import cv2
import numpy as np
import pywt
from scipy.linalg import hessenberg, svd
import torch
import torch.nn.functional as F
import multiprocessing as mp
import time
import os
import json
import pickle
from noise_utils import apply_noise_attack


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA 사용")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS 사용")
    else:
        device = torch.device('cpu')
        print("GPU 사용 불가")
    return device

device = setup_device()

def resize_image(image, size=(256, 256)):
    if image.shape[:2] != size:
        return cv2.resize(image, size)
    return image

def rdwt_decompose(image, wavelet='haar'):
    coeffs = pywt.swt2(image, wavelet, level=1, trim_approx=False)
    cA, (cH, cV, cD) = coeffs[0]
    return cA, cH, cV, cD

def rdwt_reconstruct(cA, cH, cV, cD, wavelet='haar'):
    coeffs = [(cA, (cH, cV, cD))]
    reconstructed = pywt.iswt2(coeffs, wavelet)
    return reconstructed

def save_alpha_values(alpha_dict, filepath):
    try:
        alpha_str_keys = {str(k): v for k, v in alpha_dict.items()}
        
        with open(filepath, 'w') as f:
            json.dump(alpha_str_keys, f, indent=2)
        
        return True
    
    except Exception as e:
        return False

def load_alpha_values(filepath):
    try:
        if not os.path.exists(filepath):
            return {}
        
        with open(filepath, 'r') as f:
            alpha_str_keys = json.load(f)
        
        alpha_dict = {int(k): v for k, v in alpha_str_keys.items()}
        return alpha_dict
    
    except Exception as e:
        return {}

def embed_watermark(host_image, watermark_image, alpha):
    host_image = resize_image(host_image)
    watermark_image = resize_image(watermark_image)
    
    # 1.호스트 이미지 RDWT 분해
    LL_host, LH_host, HL_host, HH_host = rdwt_decompose(host_image)
    
    # 2.LL 서브밴드 Hessenberg 분해
    min_dim = min(LL_host.shape)
    LL_square = LL_host[:min_dim, :min_dim].copy()
    
    try:
        H_host, P_host = hessenberg(LL_square, calc_q=True)
    except:
        H_host = hessenberg(LL_square, calc_q=False)
        P_host = np.eye(LL_square.shape[0])
    
    # 3.H 행렬 SVD 분해 - SH 획득
    UH, SH, VHT = svd(H_host)
    
    # 4. 워터마크 SVD 분해 - SW 획득
    UW, SW, VWT = svd(watermark_image)
    
    # 5.워터마크 삽입
    # SCW = SH + α × SW
    min_len = min(len(SH), len(SW))
    SCW = SH.copy()
    SCW[:min_len] = SH[:min_len] + alpha * SW[:min_len]
    
    # 6. 새로운 H 행렬 구성
    H_watermarked = UH @ np.diag(SCW) @ VHT
    
    # 7. 역 Hessenberg 변환
    LL_watermarked = LL_host.copy()
    LL_watermarked[:min_dim, :min_dim] = P_host @ H_watermarked @ P_host.T
    
    # 8. IRDWT 적용
    watermarked_image = rdwt_reconstruct(LL_watermarked, LH_host, HL_host, HH_host)
    watermarked_image = resize_image(watermarked_image)
    
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

def extract_watermark(watermarked_image, original_host_image, original_watermark_image, alpha):
    # 이미지 크기 정규화
    watermarked_image = resize_image(watermarked_image)
    original_host_image = resize_image(original_host_image)
    original_watermark_image = resize_image(original_watermark_image)
    
    try:
        LL_watermarked, _, _, _ = rdwt_decompose(watermarked_image)
        min_dim = min(LL_watermarked.shape)
        LL_watermarked_square = LL_watermarked[:min_dim, :min_dim]
        
        try:
            H_watermarked, P_watermarked = hessenberg(LL_watermarked_square, calc_q=True)
        except:
            H_watermarked = hessenberg(LL_watermarked_square, calc_q=False)
            P_watermarked = np.eye(LL_watermarked_square.shape[0])
        
        UHW, SHW, VHWT = svd(H_watermarked)  # SHW: 워터마크된 이미지의 특이값
        
        LL_original, _, _, _ = rdwt_decompose(original_host_image)
        LL_original_square = LL_original[:min_dim, :min_dim]
        
        try:
            H_original, P_original = hessenberg(LL_original_square, calc_q=True)
        except:
            H_original = hessenberg(LL_original_square, calc_q=False)
            P_original = np.eye(LL_original_square.shape[0])
        
        UH_original, SH, VHT_original = svd(H_original)  # SH: 원본 호스트의 특이값
        
        min_len = min(len(SHW), len(SH))
        SW_extracted = extract_singular_values(SHW[:min_len], SH[:min_len], alpha)
        
        UW_original, SW_original, VWT_original = svd(original_watermark_image)
        
        extracted_watermark = reconstruct_from_svd(
            UW_original, SW_extracted, VWT_original, original_watermark_image.shape
        )
        
    except Exception as e:
        extracted_watermark = np.zeros_like(original_watermark_image)
    
    return np.clip(extracted_watermark, 0, 255).astype(np.uint8)

def extract_singular_values(SHW, SH, alpha, tolerance=1e-8):
    SW_extracted = np.zeros_like(SHW)
    
    for i in range(len(SHW)):
        if abs(alpha) > tolerance:
            extracted = (SHW[i] - SH[i]) / alpha
            
            if extracted >= 0:
                SW_extracted[i] = min(extracted, SHW[i] * 2.0)  # 상한 제한
            else:
                SW_extracted[i] = 0
        else:
            SW_extracted[i] = 0
    
    return SW_extracted

def reconstruct_from_svd(UW_original, SW_extracted, VWT_original, target_shape):
    try:
        # 차원 검증
        min_rank = min(UW_original.shape[1], VWT_original.shape[0], len(SW_extracted))
        
        if min_rank <= 0:
            return np.zeros(target_shape)
        
        # 특이값 벡터 길이 맞추기
        SW_padded = np.zeros(min_rank)
        valid_len = min(len(SW_extracted), min_rank)
        SW_padded[:valid_len] = SW_extracted[:valid_len]
        
        # 음수 특이값 방지
        SW_padded = np.maximum(SW_padded, 0)
        
        # 올바른 SVD 재구성: A = U @ S @ V^T
        S_matrix = np.diag(SW_padded)
        reconstructed = UW_original[:, :min_rank] @ S_matrix @ VWT_original[:min_rank, :]
        
        # 크기 조정
        if reconstructed.shape != target_shape:
            reconstructed = cv2.resize(reconstructed, (target_shape[1], target_shape[0]))
        
        return reconstructed
        
    except Exception as e:
        return np.zeros(target_shape)


def calculate_nc(original_watermark, extracted_watermark):
    
    # 크기 맞추기
    if original_watermark.shape != extracted_watermark.shape:
        extracted_watermark = cv2.resize(extracted_watermark, 
                                       (original_watermark.shape[1], original_watermark.shape[0]))
    
    # 정규화된 상관계수 계산
    orig_flat = original_watermark.flatten().astype(np.float64)
    ext_flat = extracted_watermark.flatten().astype(np.float64)
    
    # 평균 제거
    orig_flat = orig_flat - np.mean(orig_flat)
    ext_flat = ext_flat - np.mean(ext_flat)
    
    # 정규화된 상관계수
    numerator = np.sum(orig_flat * ext_flat)
    denominator = np.sqrt(np.sum(orig_flat ** 2)) * np.sqrt(np.sum(ext_flat ** 2))
    
    if denominator != 0:
        nc = numerator / denominator
    else:
        nc = 0.0
    
    return max(0.0, min(1.0, nc))

def calculate_ber(original_watermark, extracted_watermark):
    if original_watermark.shape != extracted_watermark.shape:
        extracted_watermark = cv2.resize(extracted_watermark, 
                                       (original_watermark.shape[1], original_watermark.shape[0]))
    
    # 이진화 (임계값 128)
    orig_binary = (original_watermark > 128).astype(np.uint8)
    ext_binary = (extracted_watermark > 128).astype(np.uint8)
    
    # 다른 비트 개수 계산
    error_bits = np.sum(orig_binary != ext_binary)
    total_bits = orig_binary.size
    
    # BER 계산
    ber = error_bits / total_bits
    return ber

def calculate_psnr(original, watermarked):
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(original, watermarked):
    mu1 = np.mean(original)
    mu2 = np.mean(watermarked)
    
    sigma1_sq = np.var(original)
    sigma2_sq = np.var(watermarked)
    sigma12 = np.mean((original - mu1) * (watermarked - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim

class WOA:
    
    def __init__(self, n_whales=8, max_iter=5, dim=1):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.dim = dim
        self.device = device
        
        # alpha값 범위 설정
        self.positions = np.random.uniform(0.01, 0.5, (n_whales, dim))
        self.best_position = None
        self.best_fitness = float('inf')
    
    def calculate_fitness(self, alpha, host_image, watermark_image, psnr_target=40):
        try:
            # 워터마크 삽입
            watermarked = embed_watermark(host_image, watermark_image, alpha)
            
            # PSNR 계산
            psnr = calculate_psnr(host_image, watermarked)
            
            # 다양한 공격 시뮬레이션 및 NC 계산
            nc_values = []
            
            # 1. 무공격 추출 테스트
            extracted_clean = extract_watermark(watermarked, host_image, watermark_image, alpha)
            nc0 = calculate_nc(watermark_image, extracted_clean)
            nc_values.append(nc0 * 2)  # 가중치 부여
            
            # 2. 가우시안 노이즈 공격
            attacked_gaussian = self.apply_gaussian_noise(watermarked, variance=0.02)
            extracted_gaussian = extract_watermark(attacked_gaussian, host_image, watermark_image, alpha)
            nc1 = calculate_nc(watermark_image, extracted_gaussian)
            nc_values.append(nc1)
            
            # 3. Salt & Pepper 노이즈 공격
            attacked_sp = self.apply_salt_pepper_noise(watermarked, density=0.025)
            extracted_sp = extract_watermark(attacked_sp, host_image, watermark_image, alpha)
            nc2 = calculate_nc(watermark_image, extracted_sp)
            nc_values.append(nc2)
            
            # 4. 중앙값 필터 공격
            attacked_median = self.apply_median_filter(watermarked, kernel_size=5)
            extracted_median = extract_watermark(attacked_median, host_image, watermark_image, alpha)
            nc3 = calculate_nc(watermark_image, extracted_median)
            nc_values.append(nc3)
            
            # 5. 블러 공격
            attacked_blur = self.apply_blur_filter(watermarked, kernel_size=(5, 5), sigma=1.5)
            extracted_blur = extract_watermark(attacked_blur, host_image, watermark_image, alpha)
            nc4 = calculate_nc(watermark_image, extracted_blur)
            nc_values.append(nc4)
            
            # 개선된 적합도 함수
            avg_nc = np.mean(nc_values)
            fitness = 3 * abs(psnr - psnr_target) + 20 * abs(1 - avg_nc)
            
            # 품질 체크
            if psnr < 25:
                fitness += 50
            
            if avg_nc < 0.2:
                fitness += 50
            
            if nc0 < 0.3:
                fitness += 100
            
            return fitness
            
        except Exception as e:
            return 1000.0
    
    def apply_gaussian_noise(self, image, variance=0.02):
        noise = np.random.normal(0, np.sqrt(variance), image.shape)
        return np.clip(image.astype(np.float64) + noise * 255, 0, 255).astype(np.uint8)
    
    def apply_salt_pepper_noise(self, image, density=0.025):
        noisy = image.copy()
        total_pixels = image.size
        
        # Salt noise
        salt_pixels = int(total_pixels * density / 2)
        salt_coords = [np.random.randint(0, i, salt_pixels) for i in image.shape]
        noisy[tuple(salt_coords)] = 255
        
        # Pepper noise
        pepper_pixels = int(total_pixels * density / 2)
        pepper_coords = [np.random.randint(0, i, pepper_pixels) for i in image.shape]
        noisy[tuple(pepper_coords)] = 0
        
        return noisy
    
    def apply_median_filter(self, image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)
    
    def apply_blur_filter(self, image, kernel_size=(5, 5), sigma=1.5):
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def optimize_alpha(self, host_image, watermark_image):
        print(f"    WOA 최적화: {self.n_whales}마리 고래, {self.max_iter}회 반복")
        
        # 초기 평가
        for i in range(self.n_whales):
            alpha = self.positions[i, 0]
            fitness = self.calculate_fitness(alpha, host_image, watermark_image)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = self.positions[i].copy()
        
        for iteration in range(self.max_iter):
            a = 2 - iteration * (2 / self.max_iter)
            
            # 모든 고래의 적합도 평가
            for i in range(self.n_whales):
                alpha = self.positions[i, 0]
                fitness = self.calculate_fitness(alpha, host_image, watermark_image)
                
                # 최적해 업데이트
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = self.positions[i].copy()
                    print(f"      새로운 최적해: alpha={alpha:.6f}, fitness={fitness:.4f}")
            
            # 고래 위치 업데이트
            for i in range(self.n_whales):
                r1 = np.random.random()
                r2 = np.random.random()
                
                A = 2 * a * r1 - a
                C = 2 * r2 
                p = np.random.random()
                
                if p < 0.5: 
                    if abs(A) >= 1:
                        # 랜덤 탐색
                        rand_idx = np.random.randint(0, self.n_whales)
                        D = abs(C * self.positions[rand_idx] - self.positions[i])
                        self.positions[i] = self.positions[rand_idx] - A * D
                    else:
                        # 최적해 주변 탐색
                        if self.best_position is not None:
                            D = abs(C * self.best_position - self.positions[i])
                            self.positions[i] = self.best_position - A * D
                else:
                    # 나선형 업데이트
                    if self.best_position is not None:
                        D = abs(self.best_position - self.positions[i])
                        b = 1
                        l = np.random.uniform(-1, 1)
                        self.positions[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_position
                
                # 경계 조건
                self.positions[i] = np.clip(self.positions[i], 0.01, 0.5)
        
        final_alpha = self.best_position[0] if self.best_position is not None else 0.5
        print(f"    최적 alpha값: {final_alpha:.6f} (적합도: {self.best_fitness:.4f})")
        return final_alpha

def process_frame_watermarking(frame, watermark_image, alpha, add_noise=False, noise_type='none', noise_params=None):    
    # 컬러 프레임을 그레이스케일로 변환
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    # 워터마크 삽입
    watermarked_gray = embed_watermark(gray_frame, watermark_image, alpha)
    
    # 원본 크기로 복원
    if watermarked_gray.shape != gray_frame.shape:
        watermarked_gray = cv2.resize(watermarked_gray, (gray_frame.shape[1], gray_frame.shape[0]))
    
    # 컬러 프레임 복원 (YCrCb 색공간 사용)
    if len(frame.shape) == 3:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = watermarked_gray  # Y채널(밝기)만 교체
        watermarked_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        watermarked_frame = watermarked_gray
    
    # 노이즈 추가
    if add_noise and noise_type != 'none':
        if noise_params is None:
            noise_params = {}
        
        seed = noise_params.get('seed', 42)
        
        watermarked_frame = apply_noise_attack(
            watermarked_frame, 
            noise_type=noise_type, 
            seed=seed,
            **{k: v for k, v in noise_params.items() if k != 'seed'}
        )
    
    return watermarked_frame

def embed_video_watermark_with_woa(input_video_path, output_video_path, watermark_image_path, 
                                   watermark_interval=30, use_parallel=True):
    
    # 워터마크 이미지 로드
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"워터마크 이미지를 찾을 수 없습니다: {watermark_image_path}")
    
    # 워터마크를 256x256으로 조정
    watermark = resize_image(watermark)
    
    # 비디오 캡처 설정
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오를 열 수 없습니다: {input_video_path}")
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 워터마크 삽입 대상 프레임 계산
    watermark_frames = [i for i in range(watermark_interval, total_frames + 1, watermark_interval)]
    
    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    watermarked_frames_count = 0
    optimization_count = 0
    start_time = time.time()
    
    # 최적 alpha값 저장용 딕셔너리
    alpha_cache = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if (frame_idx + 1) % watermark_interval == 0:  # 0-based 인덱스이므로 +1
            watermark_frame_number = frame_idx + 1
                        
            # 현재 프레임을 기준으로 최적 alpha값 계산
            if len(frame.shape) == 3:
                sample_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                sample_gray = frame
            
            sample_resized = resize_image(sample_gray)
            
            # WOA 최적화 수행
            optimizer = WOA(n_whales=8, max_iter=5)
            optimal_alpha = optimizer.optimize_alpha(sample_resized.astype(np.float64), watermark.astype(np.float64))
            
            # 결과 저장
            alpha_cache[watermark_frame_number] = optimal_alpha
            optimization_count += 1
            
            # 워터마크 삽입 (노이즈 없음)
            watermarked_frame = process_frame_watermarking(
                frame, watermark, optimal_alpha, 
                add_noise=False, noise_type='none'
            )
            watermarked_frames_count += 1
            
            print(f"프레임 {watermark_frame_number}: alpha = {optimal_alpha:.6f} 적용")
            
        else:
            # 워터마크 없는 원본 프레임
            watermarked_frame = frame
        
        # 프레임 출력
        out.write(watermarked_frame)
        frame_idx += 1
    
    # 메모리 해제
    cap.release()
    out.release()
    
    # alpha값 저장
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    alpha_file_path = os.path.join(os.path.dirname(output_video_path), f"{base_name}_alpha_values.json")
    save_alpha_values(alpha_cache, alpha_file_path)
    
    # 최종 GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    total_time = time.time() - start_time
    
    for frame_num in sorted(alpha_cache.keys()):
        alpha = alpha_cache[frame_num]
    
    return alpha_cache

def embed_video_watermark_with_preset_alpha(input_video_path, output_video_path, watermark_image_path, 
                                             alpha_values, watermark_interval=30, noise_type='none', noise_params=None):
    
    # 워터마크 이미지 로드
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"워터마크 이미지를 찾을 수 없습니다: {watermark_image_path}")
    
    # 워터마크를 256x256으로 조정
    watermark = resize_image(watermark)
    
    # 비디오 캡처 설정
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오를 열 수 없습니다: {input_video_path}")
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    watermarked_frames_count = 0
    start_time = time.time()
    
    # 노이즈 파라미터 기본값 설정
    if noise_params is None:
        noise_params = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 진행률 표시
        if frame_idx % (total_frames // 10) == 0 and frame_idx > 0:
            progress = (frame_idx / total_frames) * 100
            print(f"   진행률: {progress:.1f}% ({frame_idx}/{total_frames} 프레임)")
        
        if (frame_idx + 1) % watermark_interval == 0:
            watermark_frame_number = frame_idx + 1
            
            # 미리 계산된 alpha값 사용
            optimal_alpha = alpha_values.get(watermark_frame_number)
            if optimal_alpha is None:
                print(f"   프레임 {watermark_frame_number}의 alpha값을 찾을 수 없음, 건너뜀")
                watermarked_frame = frame
            else:
                # 노이즈 파라미터에 시드 추가
                current_noise_params = noise_params.copy()
                current_noise_params['seed'] = frame_idx
                
                # 워터마크 삽입 + 노이즈 추가
                watermarked_frame = process_frame_watermarking(
                    frame, watermark, optimal_alpha, 
                    add_noise=(noise_type != 'none'), 
                    noise_type=noise_type, 
                    noise_params=current_noise_params
                )
                watermarked_frames_count += 1
        else:
            # 워터마크 없는 원본 프레임
            watermarked_frame = frame
        
        # 프레임 출력
        out.write(watermarked_frame)
        frame_idx += 1
    
    # 메모리 해제
    cap.release()
    out.release()

def analyze_video_performance(input_video_path, watermarked_video_path, original_watermark_path, 
                              alpha_values, watermark_interval=30):
    
    # 비디오 캡처 설정
    cap_original = cv2.VideoCapture(input_video_path)
    cap_watermarked = cv2.VideoCapture(watermarked_video_path)
    
    if not cap_original.isOpened() or not cap_watermarked.isOpened():
        return None
    
    # 원본 워터마크 로드
    original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    if original_watermark is None:
        return None
    
    # 워터마크를 256x256으로 조정
    original_watermark = resize_image(original_watermark)
    
    # 분석 결과 저장용
    frame_results = []
    nc_values = []
    ber_values = []
    frame_idx = 0
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    analyzed_count = 0
    
    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()
        
        if not ret_original or not ret_watermarked:
            break
        
        if (frame_idx + 1) % watermark_interval == 0:
            watermark_frame_number = frame_idx + 1
            
            # 해당 프레임의 alpha값 가져오기
            alpha = alpha_values.get(watermark_frame_number)
            if alpha is None:
                frame_idx += 1
                continue
            
            # 그레이스케일 변환
            if len(frame_original.shape) == 3:
                gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
                gray_watermarked = cv2.cvtColor(frame_watermarked, cv2.COLOR_BGR2GRAY)
            else:
                gray_original = frame_original
                gray_watermarked = frame_watermarked
            
            # 크기 조정
            gray_original = resize_image(gray_original)
            gray_watermarked = resize_image(gray_watermarked)
            
            # 워터마크 추출
            extracted_watermark = extract_watermark(
                gray_watermarked, gray_original, original_watermark, alpha
            )
            
            # NC 및 BER 계산
            nc_value = calculate_nc(original_watermark, extracted_watermark)
            ber_value = calculate_ber(original_watermark, extracted_watermark)
            
            nc_values.append(nc_value)
            ber_values.append(ber_value)
            analyzed_count += 1
            
            # 개별 프레임 결과 저장
            frame_result = {
                'frame_number': watermark_frame_number,
                'nc': nc_value,
                'ber': ber_value,
                'alpha': alpha
            }
            frame_results.append(frame_result)
        
        frame_idx += 1
    
    cap_original.release()
    cap_watermarked.release()
    
    # 결과 반환
    if nc_values and ber_values:
        analysis_results = {
            'frame_details': frame_results,
            'nc_mean': np.mean(nc_values),
            'ber_mean': np.mean(ber_values),
            'analyzed_frames': analyzed_count
        }
        return analysis_results
    else:
        return None

def analyze_all_videos():    
    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input_videos")
    output_folder = os.path.join(base_dir, "output_videos")
    watermark_image_path = os.path.join(base_dir, "watermark.png")
    
    # 모든 노이즈 타입
    noise_types = ['clean', 'gaussian', 'salt_pepper', 'blur', 'jpeg', 
                   'rotation', 'scaling', 'histogram', 'gamma', 'median']
    
    # 결과 저장용
    all_results = {}
    
    # 입력 비디오 파일들 확인
    if not os.path.exists(input_folder):
        return
    
    video_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        return
    
    # 각 노이즈 타입별로 분석
    for noise_type in noise_types:
        all_results[noise_type] = {}
        
        for video_file in video_files:
            base_name = os.path.splitext(video_file)[0]
            
            # 원본 비디오 경로
            input_video_path = os.path.join(input_folder, video_file)
            
            # 워터마크된 비디오 경로
            woa_folder = os.path.join(output_folder, "woa", f"woa_{noise_type}")
            watermarked_video_path = os.path.join(woa_folder, f"{base_name}_woa_{noise_type}.avi")
            
            alpha_file_path = os.path.join(
                os.path.join(output_folder, "woa", "woa_clean"), 
                f"{base_name}_alpha_values.json"
            )
            
            # 파일 존재 확인
            if not os.path.exists(watermarked_video_path):
                print(f"   {base_name}: 워터마크된 비디오를 찾을 수 없음")
                all_results[noise_type][base_name] = None
                continue
            
            # alpha값 로드
            alpha_values = load_alpha_values(alpha_file_path)
            if not alpha_values:
                print(f"   {base_name}: alpha값을 로드할 수 없음, 분석 건너뜀")
                all_results[noise_type][base_name] = None
                continue
            
            # 분석 수행
            analysis_result = analyze_video_performance(
                input_video_path, 
                watermarked_video_path, 
                watermark_image_path,
                alpha_values,
                watermark_interval=30
            )
            
            all_results[noise_type][base_name] = analysis_result
                
    # 전체 결과 요약 출력
    print_results(all_results, noise_types)

def print_results(all_results, noise_types):
    
    for noise_type in noise_types:
        if noise_type in all_results:
            valid_results = [r for r in all_results[noise_type].values() if r is not None]
            
            if valid_results:
                avg_nc = np.mean([r['nc_mean'] for r in valid_results])
                avg_nc_std = np.mean([r['nc_std'] for r in valid_results])
                avg_ber = np.mean([r['ber_mean'] for r in valid_results])
                avg_ber_std = np.mean([r['ber_std'] for r in valid_results])
                
                print(f"{noise_type.upper():<12} {avg_nc:<10.4f} {avg_nc_std:<10.4f} {avg_ber:<10.4f} {avg_ber_std:<10.4f}")
            else:
                print(f"{noise_type.upper():<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # 최고 성능 노이즈 타입 찾기
    print(f"\n최고 성능:")
    best_nc_type = None
    best_nc_value = 0
    best_ber_type = None  
    best_ber_value = 1
    
    for noise_type in noise_types:
        if noise_type in all_results:
            valid_results = [r for r in all_results[noise_type].values() if r is not None]
            if valid_results:
                avg_nc = np.mean([r['nc_mean'] for r in valid_results])
                avg_ber = np.mean([r['ber_mean'] for r in valid_results])
                
                if avg_nc > best_nc_value:
                    best_nc_value = avg_nc
                    best_nc_type = noise_type
                    
                if avg_ber < best_ber_value:
                    best_ber_value = avg_ber
                    best_ber_type = noise_type
    
    if best_nc_type:
        print(f"   최고 NC: {best_nc_type.upper()} (평균 {best_nc_value:.4f})")
    if best_ber_type:
        print(f"   최저 BER: {best_ber_type.upper()} (평균 {best_ber_value:.4f})")

def run_all_video_watermarking():
    # 파일 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input_videos")
    output_folder = os.path.join(base_dir, "output_videos") 
    watermark_path = os.path.join(base_dir, "watermark.png")
    
    # WOA 출력 폴더 구조 생성
    woa_base_folder = os.path.join(output_folder, "woa")
    woa_folders = {
        'clean': os.path.join(woa_base_folder, "woa_clean"),
        'gaussian': os.path.join(woa_base_folder, "woa_gaussian"),
        'salt_pepper': os.path.join(woa_base_folder, "woa_salt_pepper"),
        'blur': os.path.join(woa_base_folder, "woa_blur"),
        'jpeg': os.path.join(woa_base_folder, "woa_jpeg"),
        'rotation': os.path.join(woa_base_folder, "woa_rotation"),
        'scaling': os.path.join(woa_base_folder, "woa_scaling"),
        'histogram': os.path.join(woa_base_folder, "woa_histogram"),
        'gamma': os.path.join(woa_base_folder, "woa_gamma"),
        'median': os.path.join(woa_base_folder, "woa_median")
    }
    
    # 모든 출력 폴더 생성
    for folder in woa_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # 노이즈 파라미터 정의
    noise_configs = {
        'clean': {'noise_type': 'none'},
        'gaussian': {'noise_type': 'gaussian', 'std': 3, 'mean': 0},
        'salt_pepper': {'noise_type': 'salt_pepper', 'noise_ratio': 0.02},
        'blur': {'noise_type': 'blur', 'kernel_size': (5, 5), 'sigma': 1.5},
        'jpeg': {'noise_type': 'jpeg', 'quality': 60},
        'rotation': {'noise_type': 'rotation', 'angle': 2},
        'scaling': {'noise_type': 'scaling', 'scale_factor': 0.95},
        'histogram': {'noise_type': 'histogram'},
        'gamma': {'noise_type': 'gamma', 'gamma': 0.85},
        'median': {'noise_type': 'median', 'kernel_size': 3}
    }
    
    try:
        if os.path.exists(input_folder):
            video_files = [f for f in os.listdir(input_folder) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            if not video_files:
                return
            
            total_videos = len(video_files)
            
            for video_idx, video_file in enumerate(video_files, 1):
                input_path = os.path.join(input_folder, video_file)
                base_name = os.path.splitext(video_file)[0]                
                output_path = os.path.join(woa_folders['clean'], f"{base_name}_woa_clean.avi")
                alpha_values = embed_video_watermark_with_woa(
                    input_video_path=input_path,
                    output_video_path=output_path, 
                    watermark_image_path=watermark_path,
                    watermark_interval=30,  
                    use_parallel=True
                )
                
                # 2단계: 저장된 alpha값을 사용하여 노이즈 버전 생성
                print("\n2단계: 저장된 alpha값으로 노이즈 버전들 생성")
                
                for noise_type, params in noise_configs.items():
                    if noise_type == 'clean':
                        continue  # clean은 이미 생성됨
                    
                    print(f"\n   {noise_type.upper()} 노이즈 버전 생성 중...")
                    output_noise_path = os.path.join(woa_folders[noise_type], f"{base_name}_woa_{noise_type}.avi")
                    
                    # 저장된 alpha값 파일 경로
                    alpha_file_path = os.path.join(woa_folders['clean'], f"{base_name}_alpha_values.json")
                    loaded_alpha_values = load_alpha_values(alpha_file_path)
                    
                    if not loaded_alpha_values:
                        print(f"   alpha값을 찾을 수 없어 {noise_type} 버전 생성 건너뜀")
                        continue
                    
                    embed_video_watermark_with_preset_alpha(
                        input_video_path=input_path,
                        output_video_path=output_noise_path,
                        watermark_image_path=watermark_path,
                        alpha_values=loaded_alpha_values,
                        watermark_interval=30,
                        noise_type=params['noise_type'],
                        noise_params={k: v for k, v in params.items() if k != 'noise_type'}
                    )
        
        # 폴더별 파일 수 확인
        print(f"\n생성된 파일 요약:")
        for noise_type, folder_path in woa_folders.items():
            if os.path.exists(folder_path):
                file_count = len([f for f in os.listdir(folder_path) if f.endswith('.avi')])
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

def run_complete_pipeline():      
    run_all_video_watermarking()
    analyze_all_videos()
    

if __name__ == "__main__":
    print("선택하세요:")
    print("1. 워터마킹 수행")
    print("2. 분석 수행") 
    print("3. (워터마킹 + 분석)")
  
    choice = input("\n선택 (1/2/3/4/5): ").strip()
    
    if choice == '1':
        run_all_video_watermarking()
    elif choice == '2':
        analyze_all_videos()
    elif choice == '3':
        run_complete_pipeline()
    else:
        print("잘못된 선택입니다.")
