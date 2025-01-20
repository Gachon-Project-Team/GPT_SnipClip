import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def check_video_quality(input_video_path, watermarked_video_path):
    cap_original = cv2.VideoCapture(input_video_path)
    cap_watermarked = cv2.VideoCapture(watermarked_video_path)

    if not cap_original.isOpened():
        raise FileNotFoundError(f"Input video '{input_video_path}' not found. Please check the file path.")
    if not cap_watermarked.isOpened():
        raise FileNotFoundError(f"Watermarked video '{watermarked_video_path}' not found. Please check the file path.")

    psnr_values = []
    ssim_values = []
    frame_idx = 0
    frame_skip = 30 

    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()

        if not ret_original or not ret_watermarked:
            break

        if frame_idx % frame_skip == 0:  # Only process every 30th frame
            psnr_values.append(psnr(frame_original, frame_watermarked))

            ssim_value = ssim(frame_original, frame_watermarked, channel_axis=2)
            ssim_values.append(ssim_value)

        frame_idx += 1

    cap_original.release()
    cap_watermarked.release()

    return np.mean(psnr_values), np.mean(ssim_values)

def plot_comparison(psnr_results, ssim_results, delta_psnr_results, delta_ssim_results, methods, output_path):
    plt.figure(figsize=(16, 18))

    x_labels = [f"video{i+1}" for i in range(len(next(iter(psnr_results['watermark_only'].values()))))]
    x_values = range(len(x_labels))

    # 1. Only Watermark PSNR
    plt.subplot(3, 2, 1)
    for method, psnr in psnr_results['watermark_only'].items():
        plt.plot(x_values, psnr, marker='o', label=method)
    plt.xticks(x_values, x_labels)
    plt.title("PSNR Comparison (Watermark Only)")
    plt.xlabel("Videos")
    plt.ylabel("PSNR (dB)")
    plt.legend()

    # 2. Only Watermark SSIM
    plt.subplot(3, 2, 2)
    for method, ssim in ssim_results['watermark_only'].items():
        plt.plot(x_values, ssim, marker='o', label=method)
    plt.xticks(x_values, x_labels)
    plt.title("SSIM Comparison (Watermark Only)")
    plt.xlabel("Videos")
    plt.ylabel("SSIM")
    plt.legend()

    # 3. Watermark+Noise PSNR
    plt.subplot(3, 2, 3)
    for method, psnr in psnr_results['watermark_with_noise'].items():
        plt.plot(x_values, psnr, marker='x', label=f"{method} (Noise)")
    plt.xticks(x_values, x_labels)
    plt.title("PSNR Comparison (Watermark+Noise)")
    plt.xlabel("Videos")
    plt.ylabel("PSNR (dB)")
    plt.legend()

    # 4. Watermark+Noise SSIM
    plt.subplot(3, 2, 4)
    for method, ssim in ssim_results['watermark_with_noise'].items():
        plt.plot(x_values, ssim, marker='x', label=f"{method} (Noise)")
    plt.xticks(x_values, x_labels)
    plt.title("SSIM Comparison (Watermark+Noise)")
    plt.xlabel("Videos")
    plt.ylabel("SSIM")
    plt.legend()

    # 5. Delta PSNR
    plt.subplot(3, 2, 5)
    bar_width = 0.2
    for i, method in enumerate(methods):
        plt.bar([x + i * bar_width for x in x_values], delta_psnr_results[method], bar_width, label=method)
    plt.xticks([x + bar_width for x in x_values], x_labels)
    plt.title("Delta PSNR ((Only Watermark) - (Watermark+Noise))")
    plt.xlabel("Videos")
    plt.ylabel("Delta PSNR")
    plt.legend()

    # 6. Delta SSIM
    plt.subplot(3, 2, 6)
    for i, method in enumerate(methods):
        plt.bar([x + i * bar_width for x in x_values], delta_ssim_results[method], bar_width, label=method)
    plt.xticks([x + bar_width for x in x_values], x_labels)
    plt.title("Delta SSIM ((Only Watermark) - (Watermark+Noise))")
    plt.xlabel("Videos")
    plt.ylabel("Delta SSIM")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":

    # 파일 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "input_videos")
    dct_folder = os.path.join(base_dir, "output_videos", "dct")
    dct_noise_folder = os.path.join(base_dir, "output_videos", "dct_with_noise")
    dwt_folder = os.path.join(base_dir, "output_videos", "dwt")
    dwt_noise_folder = os.path.join(base_dir, "output_videos", "dwt_with_noise")
    watermark_only_folder = os.path.join(base_dir, "output_videos", "watermark_only")
    watermark_with_noise_folder = os.path.join(base_dir, "output_videos", "watermark_with_noise")

    methods = ["dct", "dwt", "dwt-dct"]

    psnr_results = {'watermark_only': {method: [] for method in methods},
                    'watermark_with_noise': {method: [] for method in methods}}
    ssim_results = {'watermark_only': {method: [] for method in methods},
                    'watermark_with_noise': {method: [] for method in methods}}
    delta_psnr_results = {method: [] for method in methods}
    delta_ssim_results = {method: [] for method in methods}

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            input_video_path = os.path.join(input_folder, filename)

            for method in methods:
                if method == "dct":
                    watermark_only_path = os.path.join(dct_folder, f"{os.path.splitext(filename)[0]}_dct.avi")
                    watermark_with_noise_path = os.path.join(dct_noise_folder, f"{os.path.splitext(filename)[0]}_dct_with_noise.avi")
                elif method == "dwt":
                    watermark_only_path = os.path.join(dwt_folder, f"{os.path.splitext(filename)[0]}_dwt.avi")
                    watermark_with_noise_path = os.path.join(dwt_noise_folder, f"{os.path.splitext(filename)[0]}_dwt_with_noise.avi")
                elif method == "dwt-dct":
                    watermark_only_path = os.path.join(watermark_only_folder, f"{os.path.splitext(filename)[0]}_watermarked.avi")
                    watermark_with_noise_path = os.path.join(watermark_with_noise_folder, f"{os.path.splitext(filename)[0]}_watermarked_with_noise.avi")

                # 1. 워터마크만 삽입된 비디오들의 PSNR, SSIM값
                psnr_only, ssim_only = check_video_quality(input_video_path, watermark_only_path)
                psnr_results['watermark_only'][method].append(psnr_only)
                ssim_results['watermark_only'][method].append(ssim_only)

                # 2. 워터마크+노이즈가 삽입된 비디오들의 PSNR,SSIM값
                psnr_noise, ssim_noise = check_video_quality(input_video_path, watermark_with_noise_path)
                psnr_results['watermark_with_noise'][method].append(psnr_noise)
                ssim_results['watermark_with_noise'][method].append(ssim_noise)

                # 3. PSNR 변화량
                delta_psnr = psnr_only - psnr_noise
                delta_psnr_results[method].append(delta_psnr)

                # 4. SSIM 변화량
                delta_ssim = ssim_only - ssim_noise
                delta_ssim_results[method].append(delta_ssim)

    output_path = os.path.join(base_dir, "output_videos", "comparison_results.png")
    plot_comparison(psnr_results, ssim_results, delta_psnr_results, delta_ssim_results, methods, output_path)
    print(f"File saved in {output_path}")
