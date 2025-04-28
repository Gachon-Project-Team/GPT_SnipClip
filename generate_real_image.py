import json
import os
import torch
import requests
import shutil
import logging
import time
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import generate_scrap
from generate_script import generate_script
import api_key
import gc
import concurrent.futures
from tqdm import tqdm
import threading

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 진행률 출력을 위한 Lock
print_lock = threading.Lock()

# 이미지 저장 디렉토리
IMAGE_SAVE_DIR = "./image"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Blip 모델 초기화
def setup_blip():
    """Blip 모델 설정"""
    start_time = time.time()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Blip: MPS 디바이스 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Blip: CUDA 디바이스 사용")
    else:
        device = torch.device("cpu")
        logging.warning("Blip: CPU 디바이스 사용")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Blip 모델 로드 완료: {elapsed_time:.2f}초 소요")
    
    return model, processor, device

def cleanup_blip(model, processor):
    #Blip 모델 메모리 해제
    start_time = time.time()
    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    elapsed_time = time.time() - start_time
    logging.info(f"Blip 모델 메모리 해제: {elapsed_time:.2f}초 소요")

def generate_image_captions(model, processor, device, image_paths, batch_size=8):
    #이미지를 캡션으로 변환하는 함수 (배치 처리)
    start_time = time.time()
    captions = []
    
    # 배치 단위로 처리
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        try:
            # 이미지 로드
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            # 입력 생성 및 디바이스 이동
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # 추론 실행
            with torch.no_grad():
                out = model.generate(**inputs)
            
            # 캡션 디코딩
            batch_captions = [processor.decode(c, skip_special_tokens=True) for c in out]
            captions.extend(batch_captions)
            
            logging.info(f"배치 {i//batch_size + 1} 캡션 생성 완료 ({len(batch_paths)}개)")
        except Exception as e:
            # 오류 발생 시 빈 캡션 추가
            logging.error(f"캡션 생성 중 오류: {e}")
            captions.extend([""] * len(batch_paths))
    
    elapsed_time = time.time() - start_time
    logging.info(f"전체 이미지 캡션 생성 완료: {len(captions)}개, {elapsed_time:.2f}초 소요")
    return captions

# GPT를 사용하여 섹션과 이미지 캡션 매칭 준비
def setup_gpt_request(section_text, captions, used_indices):
    #GPT를 사용하여 섹션과 이미지 캡션 매칭 요청 준비
    start_time = time.time()
    
    key = api_key.get_gpt_key()
    url = "https://api.openai.com/v1/chat/completions"
    
    header = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신에게는 영상 대본 중 한 섹션과 이미지를 캡션화한 데이터가 주어집니다. "
                    "당신의 역할은 대본과 가장 잘 매칭되는 이미지 캡션을 찾고 해당 캡션의 인덱스를 반환하는 것입니다. "
                    f"단, 매칭된 이미지 캡션의 인덱스가 {used_indices} 이 리스트에 있는 번호들과 중복된다면, 중복되지 않도록 재매칭 하세요. "
                    "또 아래와 같은 규칙을 지켜주세요:"
                    "1. 캡션의 인덱스는 0부터 시작합니다. index를 잘못 계산하지 않도록 주의하세요. "
                    "2. 이미지 캡션에는 특정 개체에 대한 일반화된 표현이 사용될 수 있습니다. 예를 들어 '푸바오'는 '판다'로만 표현되거나, 소설작가 한강은 '치마를 입은 여성'으로 표현됩니다. 이를 고려하세요."
                    "3. 캡션은 이미지에 있는 글자를 잘못 인식해 표현했을 수 있습니다. 이를 고려하세요."
                    "4. '대본과 잘 매칭되는 이미지'란, 대본을 읽을 때 시각적 자료로 사용할 수 있을만한 즉, 대본을 이해하는데 도움을 줄만한 이미지를 의미합니다."
                    "5. 실제 세계를 찍은 이미지가 아니라 포토샵으로 만든 것같은 이미지에 대한 캡션은 매칭을 자제해주세요. 이미지 캡션엔 기사 로고 이미지에 대한 캡션이 섞여있을 수 있으니 주의해주세요."
                    "6. 캡션 인덱스 이외의 추가 설명은 반환하지 마세요. 인덱스 숫자만 반환하세요."
                    "7. a close up of a newspaper with a news paper on it 혹은 이와 유사한 캡션은 뉴스 로고에 대한 캡션이니 어떤 대본과도 매칭시키지 마세요."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is the news script: {section_text}\n"
                    f"Here are the image captions: {captions}\n"
                    f"{used_indices}에 있는 번호와 중복된 번호를 반환하지 않도록 매칭을 진행하세요.\n"
                    "Output example: 3"
                )
            }
        ]
    }
    
    elapsed_time = time.time() - start_time
    logging.info(f"GPT 요청 준비 완료: {elapsed_time:.2f}초 소요")
    
    return url, header, request

# GPT 응답 처리
def clean_gpt_response(raw_content):
    # GPT 응답 정리
    raw_content = raw_content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]  # ```json 제거
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]  # ``` 제거
    return raw_content

# GPT API 실행
def execute_gpt(url, header, request):
    # GPT API 호출 및 결과 반환
    start_time = time.time()
    
    try:
        # GPT API 호출
        response = requests.post(url, headers=header, json=request)
        
        # 응답 상태 확인
        if response.status_code == 200:
            raw_content = response.json()["choices"][0]["message"]["content"]
            clean_content = clean_gpt_response(raw_content)
            
            # 숫자만 반환
            try:
                result = int(clean_content)
                elapsed_time = time.time() - start_time
                logging.info(f"GPT 응답 성공적으로 처리: 인덱스 {result}, {elapsed_time:.2f}초 소요")
                return result
            except ValueError:
                elapsed_time = time.time() - start_time
                logging.error(f"GPT 응답이 숫자가 아님: {clean_content}, {elapsed_time:.2f}초 소요")
                return 0
        else:
            elapsed_time = time.time() - start_time
            logging.error(f"GPT 응답 오류: {response.status_code}, {elapsed_time:.2f}초 소요")
            return 0
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"GPT API 호출 중 오류: {e}, {elapsed_time:.2f}초 소요")
        return 0

def scrap_image(query, max_images=200):
    # 이미지 URL 스크래핑 함수
    start_time = time.time()
    result = []
    
    # 1. 기본 뉴스 스크래핑
    news_scraper_start_time = time.time()
    logging.info(f"'{query}' 쿼리로 뉴스 기사 스크래핑 시작")
    news_data = generate_scrap.news_scraper(query)
    news_scraper_elapsed_time = time.time() - news_scraper_start_time
    logging.info(f"뉴스 스크래핑 완료: {news_scraper_elapsed_time:.2f}초 소요")
    
    basic_images = 0
    for category, articles in news_data.items():
        for article in articles:
            if article.get("image"):
                result.append([article["image"], article["url"]])
                basic_images += 1
    
    logging.info(f"기본 뉴스 스크래핑에서 {basic_images}개 이미지 URL 수집")
    
    # 2. 추가 이미지 소스 (execute_scrap 사용)
    try:
        execute_scrap_start_time = time.time()
        logging.info("추가 이미지 소스 스크래핑")
        additional_news = generate_scrap.execute_scrap(query)
        execute_scrap_elapsed_time = time.time() - execute_scrap_start_time
        logging.info(f"추가 뉴스 스크래핑 완료: {execute_scrap_elapsed_time:.2f}초 소요")
        
        additional_images = 0
        for n in additional_news:
            if n.get("image"):
                # 중복 URL 방지
                if not any(n["image"] == img[0] for img in result):
                    result.append([n["image"], n["url"]])
                    additional_images += 1
        
        logging.info(f"추가 소스에서 {additional_images}개 이미지 URL 수집")
    except Exception as e:
        logging.error(f"추가 이미지 스크랩 중 오류: {e}")
    
    # 최대 이미지 수 제한
    if len(result) > max_images:
        logging.info(f"수집된 이미지 URL {len(result)}개 중 {max_images}개로 제한")
        result = result[:max_images]
    
    elapsed_time = time.time() - start_time
    logging.info(f"총 {len(result)}개 이미지 URL 스크랩 완료: {elapsed_time:.2f}초 소요")
    return result

def is_similar_ssim(image_path, existing_images, threshold=0.8):
    # SSIM을 사용하여 이미지 유사도 계산
    start_time = time.time()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from piq import ssim
        
        # 새 이미지 처리
        new_image = Image.open(image_path).resize((400, 400)).convert("RGB")
        new_image = torch.tensor(np.array(new_image) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # 기존 이미지와 비교
        for existing_path in existing_images:
            existing_image = Image.open(existing_path).resize((400, 400)).convert("RGB")
            existing_image = torch.tensor(np.array(existing_image) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            # SSIM 계산
            ssim_score = ssim(new_image, existing_image, data_range=1.0).item()

            if ssim_score >= threshold:
                elapsed_time = time.time() - start_time
                logging.info(f"유사 이미지 발견 (SSIM 점수: {ssim_score:.2f}): {elapsed_time:.2f}초 소요")
                return True  # 유사 이미지 찾음

    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"SSIM 비교 중 오류: {e} ({elapsed_time:.2f}초 소요)")
    
    elapsed_time = time.time() - start_time
    return False  # 유사 이미지 없음

def process_image_batch(batch_images, save_dir, min_width, min_height, existing_images_lock):
    """이미지 배치 처리 (다운로드 및 검증)"""
    results = []
    
    for idx, img_data in enumerate(batch_images):
        try:
            url = img_data[0]
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # 이미지 확인
            img = Image.open(BytesIO(response.content)).convert("RGB")
            file_extension = img.format.lower() if img.format else "jpg"
            
            # 다양한 이미지 형식 허용
            supported_formats = ['jpeg', 'jpg', 'png', 'bmp', 'tiff', 'webp', 'gif']
            if file_extension not in supported_formats:
                continue
            
            # 이미지 크기 확인
            width, height = img.size
            if width < min_width or height < min_height:
                continue
            
            # 임시 파일명 (스레드 ID 포함)
            thread_id = threading.get_ident()
            temp_file_name = f"temp_{thread_id}_{idx}.jpg"
            save_path = os.path.join(save_dir, temp_file_name)
            
            # 이미지 임시 저장
            img.save(save_path)
            
            results.append((img_data, save_path))
        except Exception as e:
            with print_lock:
                logging.error(f"이미지 다운로드 실패: {url}, 오류: {e}")
    
    return results

def download_img_parallel(image_urls, min_width=200, min_height=200, num_workers=10):
    # 병렬 처리를 통한 이미지 다운로드
    start_time = time.time()
    
    # 다운로드된 이미지와 원본 URL 매핑을 위한 딕셔너리
    path_to_original_url = {}
    
    save_dir = IMAGE_SAVE_DIR
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # 진행 현황 및 통계를 위한 변수
    successful_downloads = []
    paths = []
    existing_images = []  # 유사성 검사를 위한 경로
    existing_images_lock = threading.Lock()  # 공유 리소스 락
    
    total_urls = len(image_urls)
    logging.info(f"이미지 다운로드 시작: 총 {total_urls}개 URL, {num_workers}개 워커 사용")
    
    # 배치 크기 계산
    batch_size = max(5, total_urls // num_workers)
    batches = [image_urls[i:i+batch_size] for i in range(0, total_urls, batch_size)]
    
    # 유사성 검사 및 최종 저장을 위한 함수
    def finalize_images(temp_results):
        nonlocal successful_downloads, paths
        
        skipped_similarity = 0
        valid_images = 0
        
        # 임시 결과를 진행 표시줄
        for idx, (img_data, temp_path) in enumerate(tqdm(temp_results, desc="이미지 후처리")):
            try:
                # SSIM 기반 유사성 검사
                # 기존 이미지와 비교하여 유사한지 확인
                is_similar = False
                with existing_images_lock:
                    if existing_images and is_similar_ssim(temp_path, existing_images, threshold=0.7):
                        is_similar = True
                
                if is_similar:
                    os.remove(temp_path)  # 유사한 이미지 삭제
                    skipped_similarity += 1
                    continue
                
                # 유사하지 않은 경우 최종 경로로 이동
                final_name = f"image_{len(paths)}.jpg"
                final_path = os.path.join(save_dir, final_name)
                shutil.move(temp_path, final_path)
                
                with existing_images_lock:
                    existing_images.append(final_path)
                    successful_downloads.append(img_data)
                    paths.append(final_path)
                    
                    # 원본 URL 매핑 저장
                    path_to_original_url[final_path] = img_data[0]
                    
                    valid_images += 1
                
            except Exception as e:
                logging.error(f"이미지 최종 처리 중 오류: {e}")
        
        return valid_images, skipped_similarity
    
    # 병렬 다운로드 실행
    temp_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {executor.submit(process_image_batch, batch, save_dir, min_width, min_height, existing_images_lock): batch for batch in batches}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(batches), desc="배치 다운로드"):
            batch_results = future.result()
            temp_results.extend(batch_results)
    
    # 진행 상태 출력
    logging.info(f"다운로드 완료: {len(temp_results)}개 이미지 임시 저장됨")
    
    # 유사성 검사 및 최종 이미지 선택
    valid_images, skipped_similarity = finalize_images(temp_results)
    
    # 결과 통계 출력
    elapsed_time = time.time() - start_time
    logging.info(f"이미지 다운로드 및 처리 완료: {elapsed_time:.2f}초 소요")
    logging.info(f"- 총 URL 수: {total_urls}")
    logging.info(f"- 다운로드 성공: {len(temp_results)}")
    logging.info(f"- 유사 이미지 제외: {skipped_similarity}")
    logging.info(f"- 최종 이미지 수: {valid_images}")
    
    return successful_downloads, paths, path_to_original_url

def create_section_groups(sections, num_groups=5):
    if not sections or num_groups <= 0:
        return []
    if len(sections) <= num_groups:
        return [[section] for section in sections]

    sections_per_group = len(sections) // num_groups
    remainder = len(sections) % num_groups

    groups, start_idx = [], 0
    for i in range(num_groups):
        count = sections_per_group + (1 if i < remainder else 0)
        end_idx = start_idx + count
        groups.append(sections[start_idx:end_idx])
        start_idx = end_idx

    return groups

def generate_real_image(script_data, query):
    # Blip 모델을 사용한 이미지 매칭 메인 함수
    start_time = time.time()
    logging.info(f"'{query}' 쿼리 실제 이미지 매칭 시작 (Blip 모델 사용)")
    
    if len(script_data) == 1 and isinstance(script_data[0], list):
        script_data = script_data[0]
    
    # 스크립트 통계
    total_sections = sum(len(create_section_groups(item.get("section", []))) for item in script_data if isinstance(item, dict))
    
    # 이미지 스크래핑 시간 측정
    scrap_start_time = time.time()
    image_urls = scrap_image(query, max_images=200)
    scrap_elapsed_time = time.time() - scrap_start_time
    logging.info(f"이미지 URL 스크래핑 완료: {scrap_elapsed_time:.2f}초 소요")
    
    if not image_urls:
        logging.warning("스크랩된 이미지 URL이 없습니다.")
        return []
    
    # 병렬 다운로드 실행 시간 측정
    download_start_time = time.time()
    num_workers = min(20, os.cpu_count() * 2)  # CPU 코어 수의 2배, 최대 20개
    image_list, image_paths, path_to_original_url = download_img_parallel(image_urls, min_width=200, min_height=200, num_workers=num_workers)
    download_elapsed_time = time.time() - download_start_time
    logging.info(f"이미지 다운로드 완료: {download_elapsed_time:.2f}초 소요")
    
    if not image_paths:
        logging.warning("다운로드된 이미지가 없습니다.")
        return []
    
    # Blip 모델 로드 시간 측정
    blip_load_start = time.time()
    model, processor, device = setup_blip()
    blip_load_elapsed = time.time() - blip_load_start
    logging.info(f"Blip 모델 로드 완료: {blip_load_elapsed:.2f}초 소요")
    
    # 이미지 캡션 생성 시간 측정
    caption_start_time = time.time()
    image_captions = generate_image_captions(model, processor, device, image_paths)
    caption_elapsed_time = time.time() - caption_start_time
    logging.info(f"이미지 캡션 생성 완료: {caption_elapsed_time:.2f}초 소요")
    
    results = []
    total_matched = 0
    total_fallbacks = 0
    total_gpt_matches = 0
    
    try:
        for item_idx, item in enumerate(script_data):
            if not isinstance(item, dict):
                continue
            
            category = item.get("category", "unknown")
            title = item.get("title", category)
            sections = item.get("section") or item.get("sections") or []
            
            if not sections:
                logging.warning(f"스크립트 항목 {item_idx+1}에 섹션 없음: {title}")
                continue
            
            category_start_time = time.time()
            logging.info(f"처리 중: 카테고리 '{category}', {len(sections)}개 섹션")
            
            section_groups = create_section_groups(sections, 5)
            used_images = set()  # 이미 사용된 이미지 경로
            used_indices = []    # 이미 사용된 이미지 인덱스
            matched_images = []
            
            for group_idx, section_group in enumerate(section_groups):
                if not section_group:
                    continue
                
                section_start_time = time.time()
                combined_text = " ".join(section_group)
                logging.info(f"섹션 그룹 {group_idx+1}/{len(section_groups)} 처리 중")
                
                # 사용 가능한 이미지 및 캡션 필터링
                available_indices = [idx for idx, path in enumerate(image_paths) if path not in used_images]
                
                if not available_indices:
                    logging.warning(f"  - 남은 이미지가 없음 (섹션 그룹 {group_idx+1})")
                    continue
                
                available_captions = [image_captions[idx] for idx in available_indices]
                
                # GPT를 사용하여 섹션과 캡션 매칭
                gpt_start_time = time.time()
                url, header, request = setup_gpt_request(combined_text, available_captions, used_indices)
                best_idx_rel = execute_gpt(url, header, request)
                gpt_elapsed_time = time.time() - gpt_start_time
                
                # 상대적 인덱스(available_indices 내 위치)를 전체 인덱스로 변환
                if 0 <= best_idx_rel < len(available_indices):
                    best_idx = available_indices[best_idx_rel]
                    best_path = image_paths[best_idx]
                    
                    used_images.add(best_path)
                    used_indices.append(best_idx)
                    
                    # 원본 URL을 결과에 저장
                    original_url = path_to_original_url.get(best_path, "")
                    matched_images.append(original_url)
                    
                    logging.info(f"  - GPT 매칭 성공: 인덱스 {best_idx}, 캡션: '{image_captions[best_idx]}'")
                    logging.info(f"  - GPT 매칭 소요 시간: {gpt_elapsed_time:.2f}초")
                    total_gpt_matches += 1
                    total_matched += 1
                else:
                    # GPT 매칭 실패 시 fallback 매칭 수행
                    logging.warning(f"  - GPT 매칭 실패, fallback 매칭 시도")
                    
                    if available_indices:
                        # 간단히 첫 번째 사용 가능한 이미지 선택
                        fallback_idx = available_indices[0]
                        fallback_path = image_paths[fallback_idx]
                        
                        used_images.add(fallback_path)
                        used_indices.append(fallback_idx)
                        
                        # 원본 URL을 결과에 저장
                        original_url = path_to_original_url.get(fallback_path, "")
                        matched_images.append(original_url)
                        
                        logging.info(f"  - Fallback 매칭: 인덱스 {fallback_idx}, 캡션: '{image_captions[fallback_idx]}'")
                        total_fallbacks += 1
                    else:
                        logging.warning(f"  - 사용 가능한 이미지가 없음, 매칭 실패")
                
                section_elapsed_time = time.time() - section_start_time
                logging.info(f"  - 섹션 그룹 처리 완료: {section_elapsed_time:.2f}초 소요")
            
            result_item = {
                "category": category,
                "title": title,
                "section": sections,
                "image": matched_images
            }
            results.append(result_item)
            
            category_elapsed_time = time.time() - category_start_time
            logging.info(f"카테고리 '{category}' 처리 완료: {len(matched_images)}개 이미지, {category_elapsed_time:.2f}초 소요")
        
        # 전체 통계
        total_images = total_matched + total_fallbacks
        if total_images > 0:
            logging.info(f"이미지 매칭 결과:")
            logging.info(f"- 총 매칭: {total_images}개")
            logging.info(f"- GPT 매칭: {total_gpt_matches}개 ({total_gpt_matches/total_images*100:.1f}%)")
            logging.info(f"- Fallback 매칭: {total_fallbacks}개 ({total_fallbacks/total_images*100:.1f}%)")
        
    finally:
        # 메모리 정리
        cleanup_start = time.time()
        cleanup_blip(model, processor)
        cleanup_elapsed = time.time() - cleanup_start
        logging.info(f"모델 메모리 해제: {cleanup_elapsed:.2f}초 소요")
    
    # 전체 실행 시간
    elapsed_time = time.time() - start_time
    logging.info(f"실제 이미지 매칭 완료 (Blip 모델): {elapsed_time:.2f}초 소요")
    
    # 시간 분석 요약
    logging.info(f"시간 분석:")
    logging.info(f"- 스크래핑: {scrap_elapsed_time:.2f}초 ({scrap_elapsed_time/elapsed_time*100:.1f}%)")
    logging.info(f"- 다운로드: {download_elapsed_time:.2f}초 ({download_elapsed_time/elapsed_time*100:.1f}%)")
    logging.info(f"- Blip 로드: {blip_load_elapsed:.2f}초 ({blip_load_elapsed/elapsed_time*100:.1f}%)")
    logging.info(f"- 캡션 생성: {caption_elapsed_time:.2f}초 ({caption_elapsed_time/elapsed_time*100:.1f}%)")
    
    return results
