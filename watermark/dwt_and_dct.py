import cv2
import numpy as np
import pywt
import os
import time
from noise_utils import apply_noise_attack

def embed_watermark_dwt(frame, watermark, alpha=0.75):
    # BGR을 YCrCb로 변환
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)

    # Y 채널에 DWT 적용
    coeffs = pywt.dwt2(y_channel, 'haar')
    cA, (cH, cV, cD) = coeffs

    # 워터마크를 저주파 성분(LL)에 삽입
    rows, cols = watermark.shape
    if rows > cA.shape[0] or cols > cA.shape[1]:
        # 워터마크가 LL 서브밴드보다 크면 크기 조정
        watermark_resized = cv2.resize(watermark, (cA.shape[1], cA.shape[0]))
    else:
        watermark_resized = watermark
    
    # 워터마크 삽입
    cA_modified = cA.copy()
    wr, wc = watermark_resized.shape
    cA_modified[:wr, :wc] += alpha * watermark_resized

    # 역 DWT 적용
    y_modified = pywt.idwt2((cA_modified, (cH, cV, cD)), 'haar')
    y_modified = np.uint8(np.clip(y_modified, 0, 255))

    # 수정된 Y채널과 Cr, Cb 채널 병합
    ycrcb_modified = cv2.merge((y_modified, cr, cb))
    frame_modified = cv2.cvtColor(ycrcb_modified, cv2.COLOR_YCrCb2BGR)

    return frame_modified

def embed_watermark_dct(frame, watermark, alpha=0.75):
    # BGR을 YCrCb로 변환
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    
    # Y 채널을 float32로 변환
    y_float = y_channel.astype(np.float32)
    
    # 전체 이미지에 DCT 적용
    dct_full = cv2.dct(y_float)
    
    # 워터마크 크기 조정 (64x64로 고정)
    watermark_resized = cv2.resize(watermark, (64, 64)).astype(np.float32)
    
    # 저주파 영역에 워터마크 삽입 (좌상단 64x64 영역)
    h, w = dct_full.shape
    wm_h, wm_w = min(64, h), min(64, w)
    
    dct_modified = dct_full.copy()
    dct_modified[:wm_h, :wm_w] += alpha * watermark_resized[:wm_h, :wm_w] / 255.0
    
    # 역 DCT 적용
    y_modified = cv2.idct(dct_modified)
    y_modified = np.clip(y_modified, 0, 255).astype(np.uint8)
    
    # 수정된 Y채널과 Cr, Cb 채널 병합
    ycrcb_modified = cv2.merge((y_modified, cr, cb))
    frame_modified = cv2.cvtColor(ycrcb_modified, cv2.COLOR_YCrCb2BGR)

    return frame_modified

def extract_watermark_dwt(watermarked_frame, original_frame, watermark_shape, alpha=0.75):
    try:
        # BGR을 YCrCb로 변환 (삽입과 동일)
        ycrcb_watermarked = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_original = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)
        
        y_watermarked = ycrcb_watermarked[:,:,0]
        y_original = ycrcb_original[:,:,0]

        # DWT 적용
        coeffs_watermarked = pywt.dwt2(y_watermarked, 'haar')
        coeffs_original = pywt.dwt2(y_original, 'haar')
        
        cA_watermarked, _ = coeffs_watermarked
        cA_original, _ = coeffs_original

        # 워터마크 추출 (LL 서브밴드에서)
        wr, wc = watermark_shape
        if wr <= cA_watermarked.shape[0] and wc <= cA_watermarked.shape[1]:
            # 삽입 공식: cA_modified = cA + alpha * watermark
            # 추출 공식: watermark = (cA_modified - cA) / alpha
            extracted_watermark = (cA_watermarked[:wr, :wc] - cA_original[:wr, :wc]) / alpha
            extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
        else:
            extracted_watermark = np.zeros(watermark_shape, dtype=np.uint8)

        return extracted_watermark
    except Exception as e:
        return np.zeros(watermark_shape, dtype=np.uint8)

def extract_watermark_dct(watermarked_frame, original_frame, watermark_shape, alpha=0.75):
    try:
        # BGR을 YCrCb로 변환 (삽입과 동일)
        ycrcb_watermarked = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_original = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)
        
        y_watermarked = ycrcb_watermarked[:,:,0].astype(np.float32)
        y_original = ycrcb_original[:,:,0].astype(np.float32)

        # 전체 이미지에 DCT 적용 (삽입과 동일)
        dct_watermarked = cv2.dct(y_watermarked)
        dct_original = cv2.dct(y_original)
        
        # 저주파 영역에서 워터마크 추출 (좌상단 64x64 영역)
        h, w = dct_watermarked.shape
        wm_h, wm_w = min(64, h), min(64, w)
        
        extracted_watermark = (dct_watermarked[:wm_h, :wm_w] - dct_original[:wm_h, :wm_w]) * 255.0 / alpha
        extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
        
        # 64x64 크기로 패딩 또는 크롭
        if extracted_watermark.shape != (64, 64):
            extracted_watermark = cv2.resize(extracted_watermark, (64, 64))
        
        # 원래 워터마크 크기로 조정
        if watermark_shape != (64, 64):
            extracted_watermark = cv2.resize(extracted_watermark, (watermark_shape[1], watermark_shape[0]))

        return extracted_watermark
    except Exception as e:
        print(f"DCT 워터마크 추출 오류: {e}")
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
    
    # 이진화
    orig_binary = (original_watermark > 128).astype(np.uint8)
    ext_binary = (extracted_watermark > 128).astype(np.uint8)
    
    # 다른 비트 개수 계산
    error_bits = np.sum(orig_binary != ext_binary)
    total_bits = orig_binary.size
    
    # BER 계산
    ber = error_bits / total_bits
    return ber

def calculate_psnr_frames(original, watermarked):
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse < 1e-8:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def analyze_watermarked_video(input_video_path, watermarked_video_path, original_watermark_path, 
                            method='dwt', alpha=0.75
                        , watermark_interval=30):

    # 비디오 캡처 설정
    cap_original = cv2.VideoCapture(input_video_path)
    cap_watermarked = cv2.VideoCapture(watermarked_video_path)
    
    if not cap_original.isOpened() or not cap_watermarked.isOpened():
        print(f" 비디오 파일을 열 수 없습니다")
        return None
    
    # 원본 워터마크 로드
    original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    if original_watermark is None:
        print(f" 워터마크 이미지를 찾을 수 없습니다: {original_watermark_path}")
        return None
    
    # 워터마크 크기를 64x64로 조정 (DCT/DWT 방식)
    original_watermark = cv2.resize(original_watermark, (64, 64))
    
    # 분석 결과 저장용
    frame_results = []
    nc_values = []
    ber_values = []
    
    frame_idx = 0
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🔍 {method.upper()} 워터마크 분석 시작: {total_frames}프레임")
    
    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()
        
        if not ret_original or not ret_watermarked:
            break
        
        if (frame_idx + 1) % watermark_interval == 0:
            watermark_frame_number = frame_idx + 1
            
            # 워터마크 추출
            if method.lower() == 'dwt':
                extracted_watermark = extract_watermark_dwt(
                    frame_watermarked, frame_original, original_watermark.shape, alpha
                )
            elif method.lower() == 'dct':
                extracted_watermark = extract_watermark_dct(
                    frame_watermarked, frame_original, original_watermark.shape, alpha
                )
            else:
                print(f" 알 수 없는 워터마킹 방법: {method}")
                return None
            
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
        print(f" 워터마크 프레임을 찾을 수 없습니다")
        return None

def print_analysis_results(analysis_results, video_name, method, noise_type):
    if not analysis_results:
        return
    
    frame_details = analysis_results['frame_details']
    
    print(f"\n {video_name} - {method.upper()}-{noise_type.upper()} 분석 결과:")
    print("-" * 60)
    
    # 각 프레임별 결과
    for frame_result in frame_details:
        print(f"   프레임 {frame_result['frame_number']:3d}: NC={frame_result['nc']:.4f}, BER={frame_result['ber']:.4f}")
    
    # 평균값
    print(f"   평균: NC={analysis_results['nc_mean']:.4f}, BER={analysis_results['ber_mean']:.4f}")

def analyze_all_dct_dwt_videos():    
    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input_videos")
    output_folder = os.path.join(base_dir, "output_videos")
    watermark_image_path = os.path.join(base_dir, "watermark.png")
    
    # 분석할 방법들과 노이즈 타입들
    methods = ['dct', 'dwt']
    noise_types = ['clean', 'gaussian', 'salt_pepper', 'blur', 'jpeg', 
                   'rotation', 'scaling', 'histogram', 'gamma', 'median']
    
    # 결과 저장용
    all_results = {}
    
    # 입력 비디오 파일들 확인
    if not os.path.exists(input_folder):
        print(f" 입력 폴더가 존재하지 않습니다: {input_folder}")
        return
    
    video_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        return

    
    # 각 방법별로 분석
    for method in methods:
        all_results[method] = {}
        
        print(f"\n {method.upper()} 방법 분석 중...")
        
        for noise_type in noise_types:
            print(f"    {noise_type.upper()} 버전 분석 중...")
            all_results[method][noise_type] = {}
            
            for video_file in video_files:
                base_name = os.path.splitext(video_file)[0]
                
                # 원본 비디오 경로
                input_video_path = os.path.join(input_folder, video_file)
                
                # 워터마크된 비디오 경로
                method_folder = os.path.join(output_folder, method, f"{method}_{noise_type}")
                watermarked_video_path = os.path.join(method_folder, f"{base_name}_{method}_{noise_type}.avi")
                
                # 파일 존재 확인
                if not os.path.exists(watermarked_video_path):
                    all_results[method][noise_type][base_name] = None
                    continue
                
                # 분석 수행
                print(f"      🔍 {base_name} 분석 중...")
                analysis_result = analyze_watermarked_video(
                    input_video_path, 
                    watermarked_video_path, 
                    watermark_image_path,
                    method=method,
                    alpha=0.2
                ,
                    watermark_interval=30
                )
                
                all_results[method][noise_type][base_name] = analysis_result
                
                # 개별 결과 출력
                print_analysis_results(analysis_result, base_name, method, noise_type)

def insert_watermark(input_video_path, output_video_path, watermark_image_path, 
                            method='dwt', watermark_interval=30, alpha=0.2
                        , 
                            add_noise=False, noise_type='none', noise_params=None):

    # 워터마크 이미지를 그레이스케일로 읽어온 후 64x64 크기로 조정
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"워터마크 이미지를 찾을 수 없습니다: {watermark_image_path}")
    
    watermark = cv2.resize(watermark, (64, 64))
    watermark = watermark.astype(np.float32) 

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
            # 워터마킹 방법 선택
            if method.lower() == 'dct':
                frame_with_watermark = embed_watermark_dct(frame, watermark, alpha)
            elif method.lower() == 'dwt':
                frame_with_watermark = embed_watermark_dwt(frame, watermark, alpha)
            else:
                raise ValueError(f"오류: {method}")

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
    print(f"총 처리 시간: {total_time:.1f}초")

def process_all_videos():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input_videos")
    output_folder = os.path.join(base_dir, "output_videos")
    watermark_image_path = os.path.join(base_dir, "watermark.png")

    # DCT 관련 폴더
    dct_base_folder = os.path.join(output_folder, "dct")
    dct_folders = {
        'clean': os.path.join(dct_base_folder, "dct_clean"),
        'gaussian': os.path.join(dct_base_folder, "dct_gaussian"),
        'salt_pepper': os.path.join(dct_base_folder, "dct_salt_pepper"),
        'blur': os.path.join(dct_base_folder, "dct_blur"),
        'jpeg': os.path.join(dct_base_folder, "dct_jpeg"),
        'rotation': os.path.join(dct_base_folder, "dct_rotation"),
        'scaling': os.path.join(dct_base_folder, "dct_scaling"),
        'histogram': os.path.join(dct_base_folder, "dct_histogram"),
        'gamma': os.path.join(dct_base_folder, "dct_gamma"),
        'median': os.path.join(dct_base_folder, "dct_median")
    }
    
    # DWT 관련 폴더
    dwt_base_folder = os.path.join(output_folder, "dwt")
    dwt_folders = {
        'clean': os.path.join(dwt_base_folder, "dwt_clean"),
        'gaussian': os.path.join(dwt_base_folder, "dwt_gaussian"),
        'salt_pepper': os.path.join(dwt_base_folder, "dwt_salt_pepper"),
        'blur': os.path.join(dwt_base_folder, "dwt_blur"),
        'jpeg': os.path.join(dwt_base_folder, "dwt_jpeg"),
        'rotation': os.path.join(dwt_base_folder, "dwt_rotation"),
        'scaling': os.path.join(dwt_base_folder, "dwt_scaling"),
        'histogram': os.path.join(dwt_base_folder, "dwt_histogram"),
        'gamma': os.path.join(dwt_base_folder, "dwt_gamma"),
        'median': os.path.join(dwt_base_folder, "dwt_median")
    }

    # 모든 출력 폴더 생성
    all_folders = list(dct_folders.values()) + list(dwt_folders.values())
    for folder in all_folders:
        os.makedirs(folder, exist_ok=True)

    # 노이즈 파라미터 정의
    noise_configs = {
        'clean': {'noise_type': 'none'},
        'gaussian': {'noise_type': 'gaussian', 'std': 12, 'mean': 0},
        'salt_pepper': {'noise_type': 'salt_pepper', 'noise_ratio': 0.15},
        'blur': {'noise_type': 'blur', 'kernel_size': (9, 9), 'sigma': 5},
        'jpeg': {'noise_type': 'jpeg', 'quality': 10},
        'rotation': {'noise_type': 'rotation', 'angle': 5},
        'scaling': {'noise_type': 'scaling', 'scale_factor': 0.9},
        'histogram': {'noise_type': 'histogram'},
        'gamma': {'noise_type': 'gamma', 'gamma': 0.3},
        'median': {'noise_type': 'median', 'kernel_size': 9}
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
                output_path = os.path.join(dct_folders[attack_type], f"{base_name}_dct_{attack_type}.avi")
                
                # 노이즈 파라미터 추출
                noise_type = config['noise_type']
                noise_params = {k: v for k, v in config.items() if k != 'noise_type'}
                
                try:
                    insert_watermark(
                        input_video_path=input_video_path,
                        output_video_path=output_path,
                        watermark_image_path=watermark_image_path,
                        method='dct',
                        watermark_interval=30,
                        alpha=0.2
                    ,
                        add_noise=(noise_type != 'none'),
                        noise_type=noise_type,
                        noise_params=noise_params
                    )
                    print(f"    DCT + {attack_type}: {base_name}_dct_{attack_type}.avi")
                except Exception as e:
                    print(f"오류 {e}")
            
            for attack_type, config in noise_configs.items():
                
                output_path = os.path.join(dwt_folders[attack_type], f"{base_name}_dwt_{attack_type}.avi")
                
                # 노이즈 파라미터 추출
                noise_type = config['noise_type']
                noise_params = {k: v for k, v in config.items() if k != 'noise_type'}
                
                try:
                    insert_watermark(
                        input_video_path=input_video_path,
                        output_video_path=output_path,
                        watermark_image_path=watermark_image_path,
                        method='dwt',
                        watermark_interval=30,
                        alpha=0.2
                    ,
                        add_noise=(noise_type != 'none'),
                        noise_type=noise_type,
                        noise_params=noise_params
                    )
                except Exception as e:
                    print(f"오류: {e}")
                
    except Exception as e:
        print(f"오류: {e}")
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
        process_all_videos()
    elif choice == '2':
        analyze_all_dct_dwt_videos()
    elif choice == '3':
        process_all_videos()
        analyze_all_dct_dwt_videos()
    else:
        print(" 잘못된 선택")
