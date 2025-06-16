import cv2
import numpy as np
import pywt
import os
import time
import matplotlib.pyplot as plt
from noise_utils import apply_noise_attack

def embed_watermark_dwt_dct(frame, watermark, alpha=0.2):
    # 프레임을 YCrCb 색 공간으로 변환
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    
    coeffs = pywt.dwt2(y_channel, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # 저주파 성분(LL)에 DCT 적용
    cA_float = np.float32(cA)
    dct_cA = cv2.dct(cA_float)
    
    # DCT 계수에 워터마크 삽입
    rows, cols = watermark.shape
    watermark_float = watermark.astype(np.float32)
    
    # 워터마크 크기가 DCT 계수보다 클 경우 조정
    if rows > dct_cA.shape[0] or cols > dct_cA.shape[1]:
        watermark_resized = cv2.resize(watermark_float, (dct_cA.shape[1], dct_cA.shape[0]))
    else:
        watermark_resized = watermark_float
    
    # 워터마크 삽입 (좌상단 저주파 영역)
    wr, wc = watermark_resized.shape
    dct_cA_modified = dct_cA.copy()
    dct_cA_modified[:wr, :wc] += alpha * watermark_resized
    
    # 워터마크 삽입 후 역 DCT(IDCT) 적용
    cA_modified = cv2.idct(dct_cA_modified)
    
    # 역 DWT(IDWT) 적용
    y_modified = pywt.idwt2((cA_modified, (cH, cV, cD)), 'haar')
    y_modified = np.uint8(np.clip(y_modified, 0, 255))
    
    # 수정된 Y채널과 Cr, Cb 채널 병합
    ycrcb_modified = cv2.merge((y_modified, cr, cb))
    frame_modified = cv2.cvtColor(ycrcb_modified, cv2.COLOR_YCrCb2BGR)
    
    return frame_modified

def extract_watermark_dwt_dct(watermarked_frame, original_frame, watermark_shape, alpha=0.2):
    try:
        # YCrCb 색공간 변환
        ycrcb_watermarked = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_original = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)
        
        y_watermarked = ycrcb_watermarked[:,:,0]
        y_original = ycrcb_original[:,:,0]
        
        # DWT 적용
        coeffs_watermarked = pywt.dwt2(y_watermarked, 'haar')
        coeffs_original = pywt.dwt2(y_original, 'haar')
        
        cA_watermarked, _ = coeffs_watermarked
        cA_original, _ = coeffs_original
        
        # DCT 적용
        dct_watermarked = cv2.dct(np.float32(cA_watermarked))
        dct_original = cv2.dct(np.float32(cA_original))
        
        # 워터마크 추출
        wr, wc = watermark_shape
        if wr <= dct_watermarked.shape[0] and wc <= dct_watermarked.shape[1]:
            extracted_watermark = (dct_watermarked[:wr, :wc] - dct_original[:wr, :wc]) / alpha
            extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
        else:
            extracted_watermark = np.zeros(watermark_shape, dtype=np.uint8)
        
        return extracted_watermark
    except:
        return np.zeros(watermark_shape, dtype=np.uint8)

def calculate_nc(original_watermark, extracted_watermark):
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
    if mse < 1e-8:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def analyze_watermarked_video(input_video_path, watermarked_video_path, original_watermark_path, 
                            alpha=0.2, watermark_interval=30):
    # 비디오 캡처 설정
    cap_original = cv2.VideoCapture(input_video_path)
    cap_watermarked = cv2.VideoCapture(watermarked_video_path)
    
    if not cap_original.isOpened() or not cap_watermarked.isOpened():
        return None
    
    # 원본 워터마크 로드
    original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    if original_watermark is None:
        return None
    
    original_watermark = cv2.resize(original_watermark, (64, 64))
    
    # 분석 결과 저장용
    frame_results = []
    nc_values = []
    ber_values = []
    
    frame_idx = 0
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
        
    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()
        
        if not ret_original or not ret_watermarked:
            break
        
        if (frame_idx + 1) % watermark_interval == 0:
            watermark_frame_number = frame_idx + 1
            
            # 워터마크 추출
            extracted_watermark = extract_watermark_dwt_dct(
                frame_watermarked, frame_original, original_watermark.shape, alpha
            )
            
            # NC 및 BER 계산
            nc_value = calculate_nc(original_watermark, extracted_watermark)
            ber_value = calculate_ber(original_watermark, extracted_watermark)
            
            nc_values.append(nc_value)
            ber_values.append(ber_value)
            
            # 개별 프레임 결과 저장
            frame_result = {
                'frame_number': watermark_frame_number,
                'nc': nc_value,
                'ber': ber_value
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
            'ber_mean': np.mean(ber_values)
        }
        return analysis_results
    else:
        return None

def print_analysis_results(analysis_results, video_name, noise_type):
    if not analysis_results:
        return
    
    frame_details = analysis_results['frame_details']
    
    # 각 프레임별 결과
    for frame_result in frame_details:
        print(f"   프레임 {frame_result['frame_number']:3d}: NC={frame_result['nc']:.4f}, BER={frame_result['ber']:.4f}")
    
    # 평균값
    print(f"   평균: NC={analysis_results['nc_mean']:.4f}, BER={analysis_results['ber_mean']:.4f}")

def insert_watermark_enhanced(input_video_path, output_video_path, watermark_image_path, 
                            watermark_interval=30, alpha=0.2, 
                            add_noise=False, noise_type='none', noise_params=None):

    # 워터마크 이미지를 그레이스케일로 읽어온 후 64x64 크기로 조정
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"워터마크 이미지를 찾을 수 없습니다: {watermark_image_path}")
    
    watermark = cv2.resize(watermark, (64, 64))
    watermark = watermark.astype(np.float32)  # 계산 정확도를 위해 float32로 변환
    
    # 비디오 캡처 설정
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"입력 비디오를 열 수 없습니다: {input_video_path}")
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 워터마크 삽입 대상 프레임 계산
    watermark_frames = [i for i in range(watermark_interval, total_frames + 1, watermark_interval)]
    print(f"워터마크 삽입 프레임: {len(watermark_frames)}개")
    
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
        
        if (frame_idx + 1) % watermark_interval == 0:
            # DWT-DCT 하이브리드 워터마킹
            frame_with_watermark = embed_watermark_dwt_dct(frame, watermark, alpha)
            
            # 노이즈 추가
            if add_noise and noise_type != 'none':
                noise_seed = frame_idx  # 프레임 인덱스를 시드로 사용
                frame_with_watermark = apply_noise_attack(
                    frame_with_watermark, 
                    noise_type=noise_type, 
                    seed=noise_seed,
                    **noise_params
                )
            
            watermarked_frames_count += 1
        else:
            frame_with_watermark = frame
        
        out.write(frame_with_watermark)
        frame_idx += 1
    
    # 메모리 해제
    cap.release()
    out.release()
    
    # 최종 통계 출력
    total_time = time.time() - start_time

def analyze_all_dwt_dct_videos():    
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
            dwt_dct_folder = os.path.join(output_folder, "dwt_dct", f"dwt_dct_{noise_type}")
            watermarked_video_path = os.path.join(dwt_dct_folder, f"{base_name}_dwt_dct_{noise_type}.avi")
            
            # 파일 존재 확인
            if not os.path.exists(watermarked_video_path):
                all_results[noise_type][base_name] = None
                continue
            
            # 분석 수행
            analysis_result = analyze_watermarked_video(
                input_video_path, 
                watermarked_video_path, 
                watermark_image_path,
                alpha=0.2,
                watermark_interval=30
            )
            
            all_results[noise_type][base_name] = analysis_result
            
            # 개별 결과 출력
            print_analysis_results(analysis_result, base_name, noise_type)
    
    # 전체 결과 요약 출력
    print_results(all_results, noise_types)

def print_results(all_results, noise_types):
    for noise_type in noise_types:
        if noise_type in all_results:
            valid_results = [r for r in all_results[noise_type].values() if r is not None]
            
            if valid_results:
                avg_nc = np.mean([r['nc_mean'] for r in valid_results])
                avg_ber = np.mean([r['ber_mean'] for r in valid_results])
                
                print(f"{noise_type.upper():<12} {avg_nc:<10.4f} {avg_ber:<10.4f}")
            else:
                print(f"{noise_type.upper():<12} {'N/A':<10} {'N/A':<10}")

def process_all_videos_dwt_dct():    
    # 현재 파일 경로를 기준으로 상대 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input_videos")
    output_folder = os.path.join(base_dir, "output_videos")
    watermark_image_path = os.path.join(base_dir, "watermark.png")
    
    dwt_dct_base_folder = os.path.join(output_folder, "dwt_dct")
    dwt_dct_folders = {
        'clean': os.path.join(dwt_dct_base_folder, "dwt_dct_clean"),
        'gaussian': os.path.join(dwt_dct_base_folder, "dwt_dct_gaussian"),
        'salt_pepper': os.path.join(dwt_dct_base_folder, "dwt_dct_salt_pepper"),
        'blur': os.path.join(dwt_dct_base_folder, "dwt_dct_blur"),
        'jpeg': os.path.join(dwt_dct_base_folder, "dwt_dct_jpeg"),
        'rotation': os.path.join(dwt_dct_base_folder, "dwt_dct_rotation"),
        'scaling': os.path.join(dwt_dct_base_folder, "dwt_dct_scaling"),
        'histogram': os.path.join(dwt_dct_base_folder, "dwt_dct_histogram"),
        'gamma': os.path.join(dwt_dct_base_folder, "dwt_dct_gamma"),
        'median': os.path.join(dwt_dct_base_folder, "dwt_dct_median")
    }
    
    # 모든 출력 폴더 생성
    for folder in dwt_dct_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # 노이즈 파라미터 정의
    noise_configs = {
        'clean': {'noise_type': 'none'},
        'gaussian': {'noise_type': 'gaussian', 'std': 10, 'mean': 0},
        'salt_pepper': {'noise_type': 'salt_pepper', 'noise_ratio': 0.1},
        'blur': {'noise_type': 'blur', 'kernel_size': (9, 9), 'sigma': 5},
        'jpeg': {'noise_type': 'jpeg', 'quality': 20},
        'rotation': {'noise_type': 'rotation', 'angle': 5},
        'scaling': {'noise_type': 'scaling', 'scale_factor': 0.9},
        'histogram': {'noise_type': 'histogram'},
        'gamma': {'noise_type': 'gamma', 'gamma': 0.5},
        'median': {'noise_type': 'median', 'kernel_size': 7}
    }
    
    try:
        if not os.path.exists(input_folder):
            return
        
        video_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            return
        
        total_videos = len(video_files)
        
        for video_idx, filename in enumerate(video_files, 1):
            input_video_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            for attack_type, config in noise_configs.items():
                
                output_path = os.path.join(dwt_dct_folders[attack_type], f"{base_name}_dwt_dct_{attack_type}.avi")
                
                # 노이즈 파라미터 추출
                noise_type = config['noise_type']
                noise_params = {k: v for k, v in config.items() if k != 'noise_type'}
                
                try:
                    insert_watermark_enhanced(
                        input_video_path=input_video_path,
                        output_video_path=output_path,
                        watermark_image_path=watermark_image_path,
                        watermark_interval=30,
                        alpha=0.2,
                        add_noise=(noise_type != 'none'),
                        noise_type=noise_type,
                        noise_params=noise_params
                    )
                except Exception as e:
                    print(f"오류: {e}")
            
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":  
    # 사용자 선택
    print("\n선택:")
    print("1. 워터마킹")
    print("2. 분석")
    print("3. 워터마킹 + 분석")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == '1':
        process_all_videos_dwt_dct()
    elif choice == '2':
        analyze_all_dwt_dct_videos()
    elif choice == '3':
        process_all_videos_dwt_dct()
        print("\n" + "="*80)
        analyze_all_dwt_dct_videos()
    else:
        print("잘못된 숫자 입력")
