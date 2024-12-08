import cv2
import numpy as np
import pywt
import os

def extract_visible_watermark(frame_original, frame_watermarked):
    # 프레임을 YCrCb 색 공간으로 변환
    ycrcb_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2YCrCb)
    y_channel_original, _, _ = cv2.split(ycrcb_original)

    ycrcb_watermarked = cv2.cvtColor(frame_watermarked, cv2.COLOR_BGR2YCrCb)
    y_channel_watermarked, _, _ = cv2.split(ycrcb_watermarked)

    # 밝기 성분(Y) 채널에 DWT 적용(원본과 워터마크 삽입된 비디오 각각 적용)
    coeffs_original = pywt.dwt2(y_channel_original, 'haar')
    cA_original, (cH_original, cV_original, cD_original) = coeffs_original

    coeffs_watermarked = pywt.dwt2(y_channel_watermarked, 'haar')
    cA_watermarked, (cH_watermarked, cV_watermarked, cD_watermarked) = coeffs_watermarked

    dct_cA_original = cv2.dct(cA_original)
    dct_cA_watermarked = cv2.dct(cA_watermarked)

    # DCT 계수 차이를 이용하여 워터마크 정보 추출
    watermark_extracted = (dct_cA_watermarked - dct_cA_original) / 0.2
    watermark_extracted = np.uint8(np.clip(watermark_extracted, 0, 255))

    return watermark_extracted

def reveal_watermark_in_video(input_video_path, watermarked_video_path, output_folder):
    cap_original = cv2.VideoCapture(input_video_path)
    cap_watermarked = cv2.VideoCapture(watermarked_video_path)

    if not cap_original.isOpened():
        raise FileNotFoundError(f"Input video '{input_video_path}' not found. Check the file path.")
    if not cap_watermarked.isOpened():
        raise FileNotFoundError(f"Watermarked video '{watermarked_video_path}' not found. Check the file path.")

    frame_idx = 0
    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()

        if not ret_original or not ret_watermarked:
            break

        # 현재 프레임에서 워터마크 추출
        watermark_visible = extract_visible_watermark(frame_original, frame_watermarked)

        # 워터마크 이미지로 저장
        output_path = os.path.join(output_folder, f"watermark_frame_{frame_idx}.png")
        cv2.imwrite(output_path, watermark_visible)
        frame_idx += 1

    cap_original.release()
    cap_watermarked.release()
    print(f"Watermark extraction completed. Extracted frames saved in '{output_folder}'")

if __name__ == "__main__":
    # 현재 스크립트 위치를 기준으로 상대 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_folder = os.path.join(base_dir, "input_videos")
    watermarked_video_folder = os.path.join(base_dir, "output_videos")
    extracted_folder = os.path.join(base_dir, "extracted_watermarks")

    # 폴더 존재 여부 확인 및 생성
    if not os.path.exists(input_video_folder):
        raise FileNotFoundError(f"Input video folder '{input_video_folder}' not found.")
    if not os.path.exists(watermarked_video_folder):
        raise FileNotFoundError(f"Watermarked video folder '{watermarked_video_folder}' not found.")
    if not os.path.exists(extracted_folder):
        os.makedirs(extracted_folder)

    input_videos = os.listdir(input_video_folder)
    watermarked_videos = os.listdir(watermarked_video_folder)

    for watermarked_video in watermarked_videos:
        if not watermarked_video.endswith(".avi"):
            continue

        input_video_path = os.path.join(input_video_folder, watermarked_video.replace("_watermarked.avi", ".mp4"))
        watermarked_video_path = os.path.join(watermarked_video_folder, watermarked_video)

        if not os.path.exists(input_video_path):
            print(f"Corresponding input video for '{watermarked_video}' not found. Skipping...")
            continue

        output_folder = os.path.join(extracted_folder, f"{os.path.splitext(watermarked_video)[0]}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            reveal_watermark_in_video(input_video_path, watermarked_video_path, output_folder)
        except FileNotFoundError as e:
            print(e)
