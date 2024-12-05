import cv2
import numpy as np
import pywt
import os
import math
from skimage.metrics import structural_similarity as ssim

def psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        # MSE 값이 0이면 PSNR 값이 무한대로 발산
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def extract_watermark(frame_original, frame_watermarked):
    # 프레임을 YCrCb 색 공간으로 변환
    ycrcb_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2YCrCb)
    y_channel_original, _, _ = cv2.split(ycrcb_original)

    ycrcb_watermarked = cv2.cvtColor(frame_watermarked, cv2.COLOR_BGR2YCrCb)
    y_channel_watermarked, _, _ = cv2.split(ycrcb_watermarked)

    # 두 개의 Y채널에 DWT 적용
    coeffs_original = pywt.dwt2(y_channel_original, 'haar')
    cA_original, (cH_original, cV_original, cD_original) = coeffs_original

    coeffs_watermarked = pywt.dwt2(y_channel_watermarked, 'haar')
    cA_watermarked, (cH_watermarked, cV_watermarked, cD_watermarked) = coeffs_watermarked

    # 각각 저주파 성분에 DCT 적용
    dct_cA_original = cv2.dct(cA_original)
    dct_cA_watermarked = cv2.dct(cA_watermarked)

    # 워터마크 추출(사이즈를 [0, 255] 범위를 갖는 8비트 정수형으로 변환)
    watermark_extracted = (dct_cA_watermarked - dct_cA_original) / 0.05
    watermark_extracted = np.uint8(np.clip(watermark_extracted, 0, 255))

    return watermark_extracted

def check_watermarked_video(input_video_path, watermarked_video_path):
    cap_original = cv2.VideoCapture(input_video_path)
    cap_watermarked = cv2.VideoCapture(watermarked_video_path)

    if not cap_original.isOpened():
        raise FileNotFoundError(f"Input video '{input_video_path}' Not found. Please check the file path.")
    if not cap_watermarked.isOpened():
        raise FileNotFoundError(f"Watermarked video '{watermarked_video_path}' Not found. Please check the file path.")

    frame_count = 0
    success_count = 0
    psnr_values = []
    ssim_values = []
    frame_idx = 0

    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()

        if not ret_original or not ret_watermarked:
            break

        # 워터마크가 삽입된 프레임과 원본 프레임 비교 성능 평가(*현재 코드에서는 30n번째 프레임들이 해당됨)
        if frame_idx % 30 == 0:
            watermark_extracted = extract_watermark(frame_original, frame_watermarked)

            # 워터마크가 정상적으로 추출됐는지 확인
            if np.mean(watermark_extracted) > 10:  # 워터마크 밝기 임계값 설정
                success_count += 1

            # PSNR 값 계산
            psnr_value = psnr(frame_original, frame_watermarked)
            psnr_values.append(psnr_value)

            # SSIM 값 계산
            ssim_value, _ = ssim(frame_original, frame_watermarked, multichannel=True, full=True, win_size=3)
            ssim_values.append(ssim_value)

        frame_count += 1
        frame_idx += 1

    cap_original.release()
    cap_watermarked.release()

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Total Frames Checked: {frame_count}")
    print(f"Frames with Detected Watermark: {success_count}")
    print(f"Watermark Detection Success Rate: {(success_count / (frame_idx // 30 + 1)) * 100:.2f}%")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    input_folder = "input_videos"
    watermarked_folder = "output_videos"

    try:
        for filename in os.listdir(input_folder):
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                print(f"Performance evaluation for '{filename}'")
                input_video_path = os.path.join(input_folder, filename)
                watermarked_video_path = os.path.join(watermarked_folder, f"{os.path.splitext(filename)[0]}_watermarked.avi")
                check_watermarked_video(input_video_path, watermarked_video_path)
                
    except FileNotFoundError as e:
        print(e)
