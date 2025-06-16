import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sys
import time

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
        raise FileNotFoundError(f"원본 비디오를 찾을 수 없습니다: {input_video_path}")
    if not cap_watermarked.isOpened():
        raise FileNotFoundError(f"워터마크된 비디오를 찾을 수 없습니다: {watermarked_video_path}")
    
    psnr_values = []
    ssim_values = []
    frame_idx = 0
    frame_skip = 100
    
    while True:
        ret_original, frame_original = cap_original.read()
        ret_watermarked, frame_watermarked = cap_watermarked.read()
        
        if not ret_original or not ret_watermarked:
            break
        
        if (frame_idx + 1) % frame_skip == 0:
            # PSNR 계산
            psnr_values.append(psnr(frame_original, frame_watermarked))
            
            # SSIM 계산
            ssim_value = ssim(frame_original, frame_watermarked, channel_axis=2)
            ssim_values.append(ssim_value)
        
        frame_idx += 1
    
    cap_original.release()
    cap_watermarked.release()
    
    return (np.mean(psnr_values) if psnr_values else 0, 
            np.mean(ssim_values) if ssim_values else 0)

def find_project_directories():
    current_dir = os.getcwd()
    search_dirs = [current_dir]
    parent_dir = os.path.dirname(current_dir)
    if parent_dir != current_dir: 
        search_dirs.append(parent_dir)
    
    input_videos_dir = None
    output_videos_dir = None
    
    for search_dir in search_dirs:
        potential_input = os.path.join(search_dir, 'input_videos')
        if os.path.exists(potential_input):
            input_videos_dir = potential_input
        
        potential_output = os.path.join(search_dir, 'output_videos')
        if os.path.exists(potential_output):
            output_videos_dir = potential_output
        
        if input_videos_dir and output_videos_dir:
            break
    
    return input_videos_dir, output_videos_dir

def get_video_files_from_folder(folder_path):
    video_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                video_files.append(os.path.join(folder_path, file))
    return sorted(video_files)

def get_watermarking_folders(output_videos_dir):
    methods = []
    noise_types = set()
    
    if not os.path.exists(output_videos_dir):
        return [], []
    
    # output_videos 내의 모든 폴더 확인
    for item in os.listdir(output_videos_dir):
        item_path = os.path.join(output_videos_dir, item)
        if os.path.isdir(item_path):
            methods.append(item)
    
    # 각 방법 폴더 내의 노이즈 타입 폴더들 확인
    for method in methods:
        method_path = os.path.join(output_videos_dir, method)
        if os.path.exists(method_path):
            for noise_folder in os.listdir(method_path):
                noise_path = os.path.join(method_path, noise_folder)
                if os.path.isdir(noise_path):
                    if noise_folder.startswith(f"{method}_"):
                        noise_type = noise_folder[len(method) + 1:]
                        noise_types.add(noise_type)
                    else:
                        noise_types.add(noise_folder)
    
    return sorted(methods), sorted(list(noise_types))

def analyze_watermarking_methods(output_videos_dir, original_video_dir):
    methods, noise_types = get_watermarking_folders(output_videos_dir)
    
    if not methods:
        raise ValueError("워터마킹 방법 폴더를 찾을 수 없습니다.")
    
    if not noise_types:
        raise ValueError("노이즈 타입 폴더를 찾을 수 없습니다.")
    
    print(f"감지된 워터마킹 방법: {methods}")
    print(f"감지된 노이즈 타입: {noise_types}")
    
    # 결과 저장용 딕셔너리
    results = {}
    
    # 원본 비디오 파일들 가져오기
    original_videos = get_video_files_from_folder(original_video_dir)
    if not original_videos:
        raise FileNotFoundError(f"원본 비디오를 찾을 수 없습니다: {original_video_dir}")
    
    print(f"원본 비디오 {len(original_videos)}개 발견")
    
    for method in methods:
        results[method] = {}
        print(f"\n{method.upper()} 방식 분석 중...")
        
        method_path = os.path.join(output_videos_dir, method)
        
        for noise_type in noise_types:
            folder_patterns = [f"{method}_{noise_type}", noise_type]
            folder_path = None
            
            for pattern in folder_patterns:
                potential_path = os.path.join(method_path, pattern)
                if os.path.exists(potential_path):
                    folder_path = potential_path
                    break
            
            if not folder_path:
                print(f"  폴더를 찾을 수 없습니다: {method}_{noise_type} 또는 {noise_type}")
                results[method][noise_type] = {'psnr': 0, 'ssim': 0}
                continue
            
            watermarked_videos = get_video_files_from_folder(folder_path)
            
            if not watermarked_videos:
                print(f"  {noise_type}: 비디오 파일 없음")
                results[method][noise_type] = {'psnr': 0, 'ssim': 0}
                continue
            
            psnr_values = []
            ssim_values = []
            
            # 각 워터마크된 비디오에 대해 품질 측정
            for watermarked_video in watermarked_videos:
                watermarked_basename = os.path.basename(watermarked_video)
                
                matched_original = None
                for original_video in original_videos:
                    original_basename = os.path.basename(original_video)
                    original_name = os.path.splitext(original_basename)[0]
                    watermarked_name = os.path.splitext(watermarked_basename)[0]
                    
                    if original_name in watermarked_name or watermarked_name in original_name:
                        matched_original = original_video
                        break
                
                if not matched_original:
                    matched_original = original_videos[0]
                
                try:
                    psnr_val, ssim_val = check_video_quality(matched_original, watermarked_video)
                    psnr_values.append(psnr_val)
                    ssim_values.append(ssim_val)
                except Exception as e:
                    print(f"    오류 발생 ({watermarked_basename}): {e}")
                    continue
            
            # 평균값 계산
            avg_psnr = np.mean(psnr_values) if psnr_values else 0
            avg_ssim = np.mean(ssim_values) if ssim_values else 0
            
            results[method][noise_type] = {
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }
            
            print(f"  {noise_type}: {len(psnr_values)}개 비디오, PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
    
    return results, methods, noise_types

def create_noise_comparison_plots(results, methods, noise_types, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(methods)]
    bar_width = 0.6
    
    # 각 노이즈 타입별로 PSNR, SSIM 그래프 생성
    for noise_type in noise_types:
        # PSNR 그래프
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        psnr_values = [results[method][noise_type]['psnr'] for method in methods]
        ssim_values = [results[method][noise_type]['ssim'] for method in methods]
        
        x_pos = np.arange(len(methods))
        method_labels = [method.upper() for method in methods]
        
        # PSNR 막대 그래프
        bars1 = ax1.bar(x_pos, psnr_values, bar_width, color=colors, alpha=0.8)
        ax1.set_xlabel('워터마킹 방식', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title(f'PSNR - {noise_type.upper()}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(method_labels)
        ax1.grid(True, alpha=0.3)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars1, psnr_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # SSIM 막대 그래프
        bars2 = ax2.bar(x_pos, ssim_values, bar_width, color=colors, alpha=0.8)
        ax2.set_xlabel('워터마킹 방식', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title(f'SSIM - {noise_type.upper()}', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(method_labels)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars2, ssim_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{noise_type}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"그래프 저장: {output_path}")

def print_detailed_results(results, methods, noise_types):
    print("\n" + "="*80)
    print("워터마킹 방법별 노이즈 타입별 성능 비교 결과")
    print("-"*80)
    
    # 각 노이즈 타입별 성능 요약
    for noise_type in noise_types:
        print(f"\n{noise_type.upper()} 노이즈:")
        print("-" * 60)
        
        noise_results = []
        for method in methods:
            psnr_val = results[method][noise_type]['psnr']
            ssim_val = results[method][noise_type]['ssim']
            noise_results.append((method, psnr_val, ssim_val))
        
        # PSNR 기준 정렬
        noise_results_psnr = sorted(noise_results, key=lambda x: x[1], reverse=True)
        # SSIM 기준 정렬
        noise_results_ssim = sorted(noise_results, key=lambda x: x[2], reverse=True)
        
        print("   PSNR 순위:")
        for rank, (method, psnr_val, ssim_val) in enumerate(noise_results_psnr, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}위"
            print(f"     {medal} {method.upper():8}: {psnr_val:6.2f}dB")
        
        print("   SSIM 순위:")
        for rank, (method, psnr_val, ssim_val) in enumerate(noise_results_ssim, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}위"
            print(f"     {medal} {method.upper():8}: {ssim_val:6.4f}")
    
    # 전체 평균 성능
    print(f"\n전체 평균 성능:")
    print("-" * 60)
    
    overall_scores = {}
    for method in methods:
        all_psnr = [results[method][noise_type]['psnr'] for noise_type in noise_types]
        all_ssim = [results[method][noise_type]['ssim'] for noise_type in noise_types]
        
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        
        overall_scores[method] = {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    # PSNR 전체 평균 순위
    psnr_ranking = sorted(overall_scores.items(), key=lambda x: x[1]['psnr'], reverse=True)
    print("\n   전체 평균 PSNR 순위:")
    for rank, (method, scores) in enumerate(psnr_ranking, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}위"
        print(f"   {medal} {method.upper():8}: {scores['psnr']:6.2f}dB")
    
    # SSIM 전체 평균 순위
    ssim_ranking = sorted(overall_scores.items(), key=lambda x: x[1]['ssim'], reverse=True)
    print("\n   전체 평균 SSIM 순위:")
    for rank, (method, scores) in enumerate(ssim_ranking, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}위"
        print(f"   {medal} {method.upper():8}: {scores['ssim']:6.4f}")

def main():
    # 자동으로 디렉토리 찾기
    input_videos_dir, output_videos_dir = find_project_directories()
    
    if not input_videos_dir:
        print("input_videos 폴더를 찾을 수 없습니다.")
        input_videos_dir = input("원본 비디오들이 있는 디렉토리 경로를 입력하세요: ").strip()
        if not os.path.exists(input_videos_dir):
            print(f"디렉토리를 찾을 수 없습니다: {input_videos_dir}")
            return
    else:
        print(f"원본 비디오 디렉토리 발견: {input_videos_dir}")
    
    if not output_videos_dir:
        print("output_videos 폴더를 찾을 수 없습니다.")
        output_videos_dir = input("워터마크된 비디오들이 있는 기본 디렉토리 경로를 입력하세요: ").strip()
        if not os.path.exists(output_videos_dir):
            print(f"디렉토리를 찾을 수 없습니다: {output_videos_dir}")
            return
    else:
        print(f"✅ 워터마크 비디오 디렉토리 발견: {output_videos_dir}")
    
    # 결과 저장 디렉토리 설정
    result_output_dir = os.path.join(os.path.dirname(output_videos_dir), 'results')
    print(f"결과 저장 디렉토리: {result_output_dir}")

    start_time = time.time()
    
    try:
        # 분석 실행
        results, methods, noise_types = analyze_watermarking_methods(output_videos_dir, input_videos_dir)
        
        # 그래프 생성
        create_noise_comparison_plots(results, methods, noise_types, result_output_dir)
        
        # 결과 출력
        print_detailed_results(results, methods, noise_types)
        
        end_time = time.time()
        print(f"\n총 소요 시간: {end_time - start_time:.2f}초")
        print(f"결과 그래프 저장 위치: {result_output_dir}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()