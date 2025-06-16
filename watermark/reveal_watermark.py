import cv2
import numpy as np
import pywt
import os

def extract_watermark_dwt(watermarked_frame, original_frame, watermark_shape=(64, 64), alpha=0.75):
    """
    DWT 기반 워터마크 추출 (삽입과 정확히 동일한 방식)
    
    Args:
        watermarked_frame: 워터마크가 삽입된 프레임
        original_frame: 원본 프레임
        watermark_shape: 워터마크 크기 (기본값: (64, 64))
        alpha: 워터마크 강도 (기본값: 0.75)
    
    Returns:
        extracted_watermark: 추출된 워터마크
    """
    try:
        # BGR을 YCrCb로 변환 (삽입과 동일한 방식)
        ycrcb_watermarked = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_original = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)
        
        y_watermarked = ycrcb_watermarked[:,:,0]
        y_original = ycrcb_original[:,:,0]

        # Y 채널에 Haar DWT 적용 (삽입과 동일)
        coeffs_watermarked = pywt.dwt2(y_watermarked, 'haar')
        coeffs_original = pywt.dwt2(y_original, 'haar')
        
        cA_watermarked, _ = coeffs_watermarked
        cA_original, _ = coeffs_original

        # LL 서브밴드에서 워터마크 추출
        # 삽입 공식: cA_modified[:wr, :wc] += alpha * watermark_resized
        # 추출 공식: watermark = (cA_modified[:wr, :wc] - cA_original[:wr, :wc]) / alpha
        wr, wc = watermark_shape
        
        if wr <= cA_watermarked.shape[0] and wc <= cA_watermarked.shape[1]:
            extracted_watermark = (cA_watermarked[:wr, :wc] - cA_original[:wr, :wc]) / alpha
            extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
        else:
            print(f"워터마크 크기가 LL 서브밴드보다 큽니다: {watermark_shape} > {cA_watermarked.shape}")
            extracted_watermark = np.zeros(watermark_shape, dtype=np.uint8)

        return extracted_watermark
    except Exception as e:
        print(f"DWT 워터마크 추출 오류: {e}")
        return np.zeros(watermark_shape, dtype=np.uint8)

def extract_watermarks_from_video(input_video_path, watermarked_video_path, output_folder, 
                                watermark_interval=100, watermark_shape=(64, 64), alpha=0.75):
    """
    비디오에서 워터마크 추출 (워터마크가 삽입된 프레임에서만)
    
    Args:
        input_video_path: 원본 비디오 경로
        watermarked_video_path: 워터마크된 비디오 경로 (dwt_clean 폴더의 파일)
        output_folder: 추출된 워터마크 저장 폴더
        watermark_interval: 워터마크 삽입 간격 (기본값: 100)
        watermark_shape: 워터마크 크기 (기본값: (64, 64))
        alpha: 워터마크 강도 (기본값: 0.75)
    """
    cap_original = cv2.VideoCapture(input_video_path)
    cap_watermarked = cv2.VideoCapture(watermarked_video_path)
    
    if not cap_original.isOpened():
        raise FileNotFoundError(f"원본 비디오를 열 수 없습니다: {input_video_path}")
    if not cap_watermarked.isOpened():
        raise FileNotFoundError(f"워터마크된 비디오를 열 수 없습니다: {watermarked_video_path}")
    
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임 수: {total_frames}")
    
    # 워터마크가 삽입된 프레임 번호들 계산
    watermark_frames = [i for i in range(watermark_interval, total_frames + 1, watermark_interval)]
    print(f"워터마크 삽입 프레임: {watermark_frames} (총 {len(watermark_frames)}개)")
    
    frame_idx = 0
    extracted_count = 0
    
    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()
        
        if not ret_original or not ret_watermarked:
            break
        
        # 워터마크가 삽입된 프레임인지 확인 (100, 200, 300, ...)
        current_frame_number = frame_idx + 1
        if current_frame_number % watermark_interval == 0:
            print(f"프레임 {current_frame_number}에서 워터마크 추출 중...")
            
            # 워터마크 추출
            extracted_watermark = extract_watermark_dwt(
                frame_watermarked, frame_original, watermark_shape, alpha
            )
            
            # 추출된 워터마크 저장
            output_path = os.path.join(output_folder, f"watermark_frame_{current_frame_number}.png")
            cv2.imwrite(output_path, extracted_watermark)
            extracted_count += 1
            
            print(f"  → 저장됨: {output_path}")
        
        frame_idx += 1
    
    cap_original.release()
    cap_watermarked.release()
    
    print(f"\n워터마크 추출 완료!")
    print(f"총 추출된 워터마크: {extracted_count}개")
    print(f"저장 위치: {output_folder}")

if __name__ == "__main__":
    # 현재 스크립트 위치를 기준으로 상대 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_folder = os.path.join(base_dir, "input_videos")  # 원본 비디오 폴더
    dwt_clean_folder = os.path.join(base_dir, "output_videos", "dwt", "dwt_clean")  # DWT 워터마크 삽입된 비디오 폴더
    extracted_folder = os.path.join(base_dir, "extracted_watermarks")
    
    # 폴더 존재 여부 확인
    if not os.path.exists(input_video_folder):
        raise FileNotFoundError(f"원본 비디오 폴더를 찾을 수 없습니다: {input_video_folder}")
    if not os.path.exists(dwt_clean_folder):
        raise FileNotFoundError(f"DWT clean 폴더를 찾을 수 없습니다: {dwt_clean_folder}")
    
    # 출력 폴더 생성
    if not os.path.exists(extracted_folder):
        os.makedirs(extracted_folder)
    
    print("🔍 DWT 워터마크 추출 시작")
    print("=" * 50)
    print(f"원본 비디오 폴더: {input_video_folder}")
    print(f"워터마크된 비디오 폴더: {dwt_clean_folder}")
    print(f"추출 결과 저장 폴더: {extracted_folder}")
    print("=" * 50)
    
    # dwt_clean 폴더의 비디오 파일들 처리
    dwt_clean_videos = [f for f in os.listdir(dwt_clean_folder) if f.endswith(".avi")]
    
    if not dwt_clean_videos:
        print("dwt_clean 폴더에서 .avi 파일을 찾을 수 없습니다.")
        exit(1)
    
    print(f"처리할 비디오 파일: {len(dwt_clean_videos)}개")
    for video in dwt_clean_videos:
        print(f"  - {video}")
    print()
    
    for dwt_video in dwt_clean_videos:
        # dwt_clean 비디오 파일명에서 원본 비디오 파일명 추출
        # 예: test1_dwt_clean.avi → test1.mp4
        base_name = dwt_video.replace("_dwt_clean.avi", "")
        input_video_name = f"{base_name}.mp4"
        
        input_video_path = os.path.join(input_video_folder, input_video_name)  # 원본 비디오
        watermarked_video_path = os.path.join(dwt_clean_folder, dwt_video)    # 워터마크된 비디오
        
        print(f"\n🎬 처리 중: {dwt_video}")
        print(f"원본: {input_video_name}")
        
        # 원본 비디오 존재 확인
        if not os.path.exists(input_video_path):
            print(f"❌ 해당하는 원본 비디오를 찾을 수 없습니다: {input_video_name}")
            continue
        
        # 개별 비디오용 출력 폴더 생성
        video_output_folder = os.path.join(extracted_folder, f"{base_name}_extracted")
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)
        
        try:
            # 워터마크 추출 실행
            extract_watermarks_from_video(
                input_video_path=input_video_path,
                watermarked_video_path=watermarked_video_path,
                output_folder=video_output_folder,
                watermark_interval=100,  # 원본 코드와 동일
                watermark_shape=(64, 64),  # 원본 코드와 동일
                alpha=0.75  # 원본 코드와 동일
            )
            print(f"✅ {base_name} 처리 완료")
            
        except Exception as e:
            print(f"❌ {base_name} 처리 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 모든 비디오 처리 완료!")
    print(f"추출된 워터마크들은 다음 위치에 저장되었습니다: {extracted_folder}")