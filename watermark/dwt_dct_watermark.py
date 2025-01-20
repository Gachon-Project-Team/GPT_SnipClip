import cv2
import numpy as np
import pywt
import os
from noise_utils import generate_noise 

def embed_watermark_dwt(frame, watermark):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)

    coeffs = pywt.dwt2(y_channel, 'haar')
    cA, (cH, cV, cD) = coeffs

    rows, cols = watermark.shape
    cA[:rows, :cols] += 1 * watermark

    y_modified = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    y_modified = np.uint8(np.clip(y_modified, 0, 255))

    ycrcb_modified = cv2.merge((y_modified, cr, cb))
    frame_modified = cv2.cvtColor(ycrcb_modified, cv2.COLOR_YCrCb2BGR)

    return frame_modified

def embed_watermark_dct(frame, watermark):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)

    dct_y = cv2.dct(np.float32(y_channel))

    rows, cols = watermark.shape
    dct_y[:rows, :cols] += 1 * watermark

    y_modified = cv2.idct(dct_y)
    y_modified = np.uint8(np.clip(y_modified, 0, 255))

    ycrcb_modified = cv2.merge((y_modified, cr, cb))
    frame_modified = cv2.cvtColor(ycrcb_modified, cv2.COLOR_YCrCb2BGR)

    return frame_modified

def insert_watermark(input_video_path, output_video_path, watermark_image_path, method, add_noise=False):
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"Watermark image '{watermark_image_path}' not found")
    watermark = cv2.resize(watermark, (64, 64))

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Input video '{input_video_path}' not found")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 30번째 프레임마다 워터마크 및 노이즈 삽입
        if frame_idx % 30 == 0:
            if method == 'dct':
                frame_with_watermark = embed_watermark_dct(frame, watermark)
            elif method == 'dwt':
                frame_with_watermark = embed_watermark_dwt(frame, watermark)

            # 노이즈 추가 여부
            if add_noise:
                noise_seed = frame_idx  # Use frame index as seed
                noise = generate_noise(frame.shape, mean=0, std=12, seed=noise_seed)
                frame_with_watermark = np.clip(frame_with_watermark.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            frame_with_watermark = frame

        out.write(frame_with_watermark)
        frame_idx += 1

    cap.release()
    out.release()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input_videos")
    output_folder = os.path.join(base_dir, "output_videos")
    watermark_image_path = os.path.join(base_dir, "watermark.png")

    dct_folder = os.path.join(output_folder, "dct")
    dwt_folder = os.path.join(output_folder, "dwt")
    dct_noise_folder = os.path.join(output_folder, "dct_with_noise")
    dwt_noise_folder = os.path.join(output_folder, "dwt_with_noise")

    # 출력 폴더 생성
    os.makedirs(dct_folder, exist_ok=True)
    os.makedirs(dwt_folder, exist_ok=True)
    os.makedirs(dct_noise_folder, exist_ok=True)
    os.makedirs(dwt_noise_folder, exist_ok=True)

    try:
        for filename in os.listdir(input_folder):
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                input_video_path = os.path.join(input_folder, filename)

                # 1. 워터마크만 삽입된 비디오 생성
                output_video_path_dct = os.path.join(dct_folder, f"{os.path.splitext(filename)[0]}_dct.avi")
                insert_watermark(input_video_path, output_video_path_dct, watermark_image_path, method='dct', add_noise=False)

                output_video_path_dwt = os.path.join(dwt_folder, f"{os.path.splitext(filename)[0]}_dwt.avi")
                insert_watermark(input_video_path, output_video_path_dwt, watermark_image_path, method='dwt', add_noise=False)

                # 2. 워터마크+노이즈 삽입된 비디오 생성
                output_video_path_dct_noise = os.path.join(dct_noise_folder, f"{os.path.splitext(filename)[0]}_dct_with_noise.avi")
                insert_watermark(input_video_path, output_video_path_dct_noise, watermark_image_path, method='dct', add_noise=True)

                output_video_path_dwt_noise = os.path.join(dwt_noise_folder, f"{os.path.splitext(filename)[0]}_dwt_with_noise.avi")
                insert_watermark(input_video_path, output_video_path_dwt_noise, watermark_image_path, method='dwt', add_noise=True)

                print(f"'{filename}' : Videos generated")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
