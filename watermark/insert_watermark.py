import cv2
import numpy as np
import pywt
import os

def embed_watermark(frame, watermark):
    # 프레임을 YCrCb 색 공간으로 변환
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)

    ''' 
    밝기 성분(Y) 채널에 DWT 적용
    -cA: 저주파 성분
    -cH: 수평 방향 고주파 성분
    -cV: 수직 방향 고주파 성분
    -cD: 대각선 방향 고주파 성분
    '''
    coeffs = pywt.dwt2(y_channel, 'haar')
    cA, (cH, cV, cD) = coeffs

    # 저주파 성분에 DCT 적용
    dct_cA = cv2.dct(cA)

    # DCT 계수에 워터마크 삽입
    rows, cols = watermark.shape
    dct_cA[:rows, :cols] += 0.2 * watermark

    # 워터마크 삽입 후 역 DCT(IDCT) 적용
    cA_modified = cv2.idct(dct_cA)

    # 역 DWT(IDWT) 적용
    y_modified = pywt.idwt2((cA_modified, (cH, cV, cD)), 'haar')
    y_modified = np.uint8(np.clip(y_modified, 0, 255))

    # 수정된 Y채널과 Cr, Cb 채널 병합
    ycrcb_modified = cv2.merge((y_modified, cr, cb))
    frame_modified = cv2.cvtColor(ycrcb_modified, cv2.COLOR_YCrCb2BGR)

    return frame_modified


def insert_watermark(input_video_path, output_video_path, watermark_image_path):
    # 워터마크 이미지를 그레이스케일로 읽어온 후 64X64 크기로 조정
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise FileNotFoundError(f"Watermark image '{watermark_image_path}' Not found")
    watermark = cv2.resize(watermark, (64, 64))

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Input video '{input_video_path}' Not found")
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

        # 30프레임 당 워터마크 삽입(*이 부분은 올리는 숏폼 동영상 fps에 따라 조정)
        if frame_idx % 30 == 0:
            frame_with_watermark = embed_watermark(frame, watermark)
        else:
            frame_with_watermark = frame

        out.write(frame_with_watermark)
        frame_idx += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    # 현재 파일 경로를 기준으로 상대 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 위치를 기준으로 경로 설정
    input_folder = os.path.join(base_dir, "input_videos")
    output_folder = os.path.join(base_dir, "output_videos")
    watermark_image_path = os.path.join(base_dir, "watermark.png")

    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        for filename in os.listdir(input_folder):
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                input_video_path = os.path.join(input_folder, filename)
                output_video_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_watermarked.avi")
                insert_watermark(input_video_path, output_video_path, watermark_image_path)
                print(f"'{filename}' watermark embedded")
    except FileNotFoundError as e:
        print(e)
