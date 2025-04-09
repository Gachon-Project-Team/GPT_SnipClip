import os
import json
import re
import requests
import torch
import shutil
import logging
import numpy as np
import gc
import time
import threading
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from generate_scrap import news_scraper, execute_scrap
from generate_script import generate_script
from flux import execute_flux, generate_prompt
import api_key

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 진행률 출력을 위한 락
print_lock = threading.Lock()

# 이미지 저장 디렉토리
IMAGE_SAVE_DIR = "./image"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Blip 모델 초기화
def setup_blip():
    # Blip 모델 설정
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
    logging.info(f"Blip 모델 로드: {elapsed_time:.2f}초 소요")
    
    return model, processor, device

def cleanup_blip(model, processor):
    # Blip 모델 메모리 해제
    start_time = time.time()
    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    elapsed_time = time.time() - start_time
    logging.info(f"Blip 모델 메모리 해제 완료: {elapsed_time:.2f}초 소요")

def generate_image_captions(model, processor, device, image_paths, batch_size=8):
    # 이미지를 캡션으로 변환하는 함수 (배치 처리)
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
    # GPT를 사용하여 섹션과 이미지 캡션 매칭 요청 준비
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

def scrap_image(query):
    start_time = time.time()
    result = []
    
    # 뉴스 데이터 스크랩
    news_scraper_start_time = time.time()
    news_data = news_scraper(query)
    news_scraper_elapsed_time = time.time() - news_scraper_start_time
    logging.info(f"뉴스 스크래핑 완료: {news_scraper_elapsed_time:.2f}초 소요")
    
    basic_images = 0
    for category, articles in news_data.items():
        for article in articles:
            if article.get("image"):
                result.append([article["image"], article["url"]])
                basic_images += 1
    
    logging.info(f"기본 뉴스 스크래핑에서 {basic_images}개 이미지 URL 수집")
    
    # 추가 이미지 소스 (execute_scrap 사용)
    try:
        execute_scrap_start_time = time.time()
        additional_news = execute_scrap(query)
        execute_scrap_elapsed_time = time.time() - execute_scrap_start_time
        logging.info(f"추가 뉴스 스크래핑 완료: {execute_scrap_elapsed_time:.2f}초 소요")
        
        additional_images = 0
        for n in additional_news:
            if n.get("image"):
                result.append([n["image"], n["url"]])
                additional_images += 1
        
        logging.info(f"추가 소스에서 {additional_images}개 이미지 URL 수집")
    except Exception as e:
        logging.error(f"추가 이미지 스크랩 중 오류: {e}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"총 {len(result)}개 이미지 URL 스크랩 완료: {elapsed_time:.2f}초 소요")
    return result

def is_similar_ssim(image_path, existing_images, threshold=0.8):
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
    if len(existing_images) > 0:
        logging.info(f"유사 이미지 없음 (비교 대상: {len(existing_images)}개): {elapsed_time:.2f}초 소요")
    return False  # 유사 이미지 없음

def get_existing_images():
    start_time = time.time()
    existing_images = []
    if os.path.exists(IMAGE_SAVE_DIR):
        for filename in os.listdir(IMAGE_SAVE_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif')):
                existing_images.append(os.path.join(IMAGE_SAVE_DIR, filename))
    
    elapsed_time = time.time() - start_time
    logging.info(f"기존 이미지 {len(existing_images)}개 발견: {elapsed_time:.2f}초 소요")
    return existing_images

def download_img(image_urls, min_width=200, min_height=200, preserve_existing=True):
    start_time = time.time()
    
    # 다운로드된 이미지와 원본 URL 매핑을 위한 딕셔너리
    path_to_original_url = {}
    
    # 기존 이미지 폴더 체크
    existing_images = []
    if os.path.exists(IMAGE_SAVE_DIR) and preserve_existing:
        existing_images = get_existing_images()
        if existing_images:
            logging.info(f"기존 이미지 폴더에서 {len(existing_images)}개 이미지 발견")
            # 이미 이미지가 있으면 바로 반환
            if existing_images:
                # 이미지 URL 정보도 함께 만들어줘야 함 (추정)
                image_list = []
                for i, img_path in enumerate(existing_images):
                    # [이미지 URL, 출처 URL] 형태로 가정
                    # 기존 파일에 대한 원본 URL 정보가 없으므로 파일 경로 사용
                    # 실제 구현에서는 이전에 저장해둔 매핑 정보를 사용하는 것이 좋음
                    image_list.append([f"file://{img_path}", ""])
                    path_to_original_url[img_path] = f"file://{img_path}"
                
                elapsed_time = time.time() - start_time
                logging.info(f"기존 이미지 사용: {elapsed_time:.2f}초 소요")
                return image_list, existing_images, path_to_original_url
        else:
            logging.info("기존 이미지 폴더가 비어 있습니다. 새로 다운로드합니다.")
    else:
        # 디렉토리가 없으면 생성
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        logging.info(f"이미지 디렉토리 생성: {IMAGE_SAVE_DIR}")

    # 기존 폴더가 비어있거나 preserve_existing=False일 경우 새로 다운로드
    idx = 0
    successful_downloads = []  # 다운로드 성공한 이미지 저장 리스트
    existing_images = []  # 기존 저장된 이미지 경로 리스트
    
    # 다운로드 통계 트래킹
    total_urls = len(image_urls)
    skipped_format = 0
    skipped_size = 0
    skipped_similarity = 0
    failed_downloads = 0
    
    logging.info(f"총 {total_urls}개 이미지 다운로드 시작")

    for i in image_urls:
        download_start_time = time.time()
        try:
            url = i[0]
            response = requests.get(url, timeout=15)  
            response.raise_for_status()
            
            img_data = Image.open(BytesIO(response.content)).convert("RGB")
            file_extension = img_data.format.lower() if img_data.format else "jpg"

            supported_formats = ['jpeg', 'jpg', 'png', 'bmp', 'tiff', 'webp', 'gif']
            if file_extension not in supported_formats:
                logging.warning(f"지원되지 않는 이미지 형식: {file_extension}, URL: {url}")
                skipped_format += 1
                continue

            # 이미지 크기 확인
            width, height = img_data.size
            if width < min_width or height < min_height:
                logging.info(f"이미지 크기가 너무 작음 (가로: {width}, 세로: {height}), URL: {url}")
                skipped_size += 1
                continue
            
            # 파일 저장 경로 생성
            file_name = f"image_{idx}.{file_extension}"
            save_path = os.path.join(IMAGE_SAVE_DIR, file_name)

            # 이미지 임시 저장
            img_data.save(save_path)

            # SSIM 비교: 기존 이미지와 유사한지 확인
            if existing_images and is_similar_ssim(save_path, existing_images, threshold=0.8):
                logging.info(f"기존 이미지와 너무 유사함: {url}")
                os.remove(save_path)  # 유사하면 저장된 파일 삭제
                skipped_similarity += 1
                continue

            # 유사하지 않거나 첫 이미지인 경우 저장
            existing_images.append(save_path)
            successful_downloads.append(i)
            
            # 원본 URL 매핑 저장
            path_to_original_url[save_path] = url
            
            download_elapsed_time = time.time() - download_start_time  
            logging.info(f"이미지 저장 성공: {save_path} ({download_elapsed_time:.2f}초 소요)")
            idx += 1
        
        except Exception as e:
            download_elapsed_time = time.time() - download_start_time
            logging.error(f"이미지 다운로드 실패: {url}, 오류: {e} ({download_elapsed_time:.2f}초 소요)")
            failed_downloads += 1
    
    # 다운로드 통계 출력
    elapsed_time = time.time() - start_time
    logging.info(f"이미지 다운로드 완료 통계 ({elapsed_time:.2f}초 소요):")
    logging.info(f"- 총 시도: {total_urls}")
    logging.info(f"- 성공: {len(successful_downloads)}")
    logging.info(f"- 형식 오류 제외: {skipped_format}")
    logging.info(f"- 크기 제한 제외: {skipped_size}")
    logging.info(f"- 유사 이미지 제외: {skipped_similarity}")
    logging.info(f"- 실패: {failed_downloads}")
    
    return successful_downloads, existing_images, path_to_original_url

def detect_entities_with_nlp(text):
    start_time = time.time()
    try:
        # 주요 인물, 장소, 조직 등을 나타내는 패턴
        patterns = [
            r'[가-힣]{1,2} (대통령|총리|장관|의원|위원장)',  # 정치인
            r'[가-힣]+ (회장|사장|CEO|대표)',  # 기업인
            r'[가-힣A-Za-z\s]+(주식회사|기업|그룹|Corp\.|Inc\.)',  # 회사
            r'[가-힣]+(산|공원|타워|빌딩|광장|역|대학교|대학)',  # 장소
            r'[0-9]{4}년(부터|까지)?',  # 연도 언급
            r'서울|부산|인천|대구|광주|대전|울산|세종|경기도|강원도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도|제주도',  # 지역명
            r'[가-힣]+시(청)?|[가-힣]+구(청)?|[가-힣]+동',  # 행정구역
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                elapsed_time = time.time() - start_time
                logging.info(f"고유명사 패턴 감지됨: {pattern} ({elapsed_time:.2f}초 소요)")
                return True
        
        elapsed_time = time.time() - start_time
        logging.info(f"고유명사 패턴 감지 안됨 ({elapsed_time:.2f}초 소요)")        
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"엔티티 감지 중 오류: {e} ({elapsed_time:.2f}초 소요)")
        return False

def determine_image_type(section, prompt):
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
                    "You are a content assistant who determines whether to use a real image or an AI-generated image based on strict criteria.\n\n"
                    "CRITERIA FOR USING REAL IMAGES (return 'real'):\n"
                    "1. Text contains names of real people (celebrities, politicians, public figures, etc.)\n"
                    "2. Text contains proper nouns (specific places, buildings, landmarks, events, organizations, etc.)\n"
                    "3. Text references specific branded products or companies that need accurate visual representation\n"
                    "4. Text describes specific historical events or moments that should be depicted accurately\n\n"
                    
                    "CRITERIA FOR USING AI IMAGES (return 'ai'):\n"
                    "1. Text describes general concepts, emotions, or abstract ideas\n"
                    "2. Text references generic objects, scenes or settings without specifying real entities\n"
                    "3. Text describes fictional scenarios or hypothetical situations\n"
                    "4. Text contains generic descriptions that don't require specific real-world representations\n\n"
                    
                    "Your response must be ONLY 'real' or 'ai' - no explanation or additional text."
                )
            },
            {
                "role": "user",
                "content": f"Text section: {section}\nAI image prompt: {prompt}"
            }
        ]
    }

    try:
        response = requests.post(url, headers=header, json=request)
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip().lower()
            
            # 로깅을 추가하여 결정 과정을 추적
            elapsed_time = time.time() - start_time
            logging.info(f"이미지 타입 결정: {'실제' if result == 'real' else 'AI'} ({elapsed_time:.2f}초 소요)")
            logging.info(f"- 섹션 미리보기: {section[:100]}...")
            return result == "real"
        else:
            elapsed_time = time.time() - start_time
            logging.error(f"GPT 응답 오류: {response.status_code} ({elapsed_time:.2f}초 소요)")
            # 에러 발생 시 기본적으로 AI 이미지 사용
            return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"이미지 타입 결정 중 오류: {e} ({elapsed_time:.2f}초 소요)")
        # 예외 발생 시 기본적으로 AI 이미지 사용
        return False

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

def generate_mix_image(script_data, query, preserve_existing=True):
    start_time = time.time()
    
    if len(script_data) == 1 and isinstance(script_data[0], list):
        script_data = script_data[0]
    
    results = []
    total_sections = 0
    total_real_images = 0
    total_ai_images = 0
    
    # 이미지 URL 스크랩 시간 측정
    scrap_start_time = time.time()
    image_urls = scrap_image(query)
    scrap_elapsed_time = time.time() - scrap_start_time
    logging.info(f"이미지 URL 스크랩 완료: {scrap_elapsed_time:.2f}초 소요")
    
    if not image_urls:
        logging.warning("스크랩된 이미지가 없습니다.")
    
    # 이미지 다운로드 시간 측정    
    download_start_time = time.time()
    image_list, image_paths, path_to_original_url = download_img(image_urls, min_width=200, min_height=200, preserve_existing=preserve_existing)
    download_elapsed_time = time.time() - download_start_time
    logging.info(f"이미지 다운로드 완료: {download_elapsed_time:.2f}초 소요")
    
    if not image_list:
        logging.warning("다운로드된 이미지가 없습니다. AI 이미지만 사용합니다.")
    
    # BLIP 모델 로드 및 캡션 생성 (실제 이미지가 있는 경우에만)
    blip_load_time = 0
    caption_time = 0
    image_captions = []
    
    if image_paths:
        # BLIP 모델 로드
        blip_start_time = time.time()
        model, processor, device = setup_blip()
        blip_load_time = time.time() - blip_start_time
        logging.info(f"BLIP 모델 로드 완료: {blip_load_time:.2f}초 소요")
        
       # 이미지 캡션 생성
        caption_start_time = time.time()
        image_captions = generate_image_captions(model, processor, device, image_paths)
        caption_time = time.time() - caption_start_time
        logging.info(f"이미지 캡션 생성 완료: {caption_time:.2f}초 소요")
    else:
        model, processor, device = None, None, None
    
    # 섹션별 이미지 매칭 시간 측정
    matching_start_time = time.time()
    
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
                
            # 섹션 그룹화
            section_groups = create_section_groups(sections, 5)
            matched_images = []
            image_types = []  # 이미지 타입 추적 (내부 로깅용)
            used_indices = []  # 이미 사용된 이미지 인덱스
            
            # 각 섹션 그룹에 대한 이미지 매칭
            for group_idx, section_group in enumerate(section_groups):
                if not section_group:
                    continue
                
                section_start_time = time.time()    
                total_sections += 1
                combined_text = " ".join(section_group)
                
                # 프롬프트 생성 시간 측정
                prompt_start_time = time.time()
                prompt = generate_prompt(combined_text)
                prompt_elapsed_time = time.time() - prompt_start_time
                logging.info(f"섹션 그룹 {group_idx+1}/{len(section_groups)} 프롬프트 생성: {prompt_elapsed_time:.2f}초 소요")
                logging.info(f"생성된 영어 프롬프트: {prompt}")
                
                # 실제 인물/고유명사 탐지 시간 측정
                entity_start_time = time.time()
                contains_entities = detect_entities_with_nlp(combined_text)
                entity_elapsed_time = time.time() - entity_start_time
                logging.info(f"엔티티 탐지 완료: {entity_elapsed_time:.2f}초 소요")
                
                # 이미지 타입 결정 시간 측정
                decision_start_time = time.time()
                # 엔티티가 명확히 감지된 경우 실제 이미지 사용
                if contains_entities and image_list:
                    use_real_image = True
                    decision_method = "엔티티 탐지"
                # 아니면 GPT로 판단
                elif image_list:  # 실제 이미지가 있는 경우만 GPT로 판단
                    use_real_image = determine_image_type(combined_text, prompt)
                    decision_method = "GPT 결정"
                else:  # 실제 이미지가 없으면 무조건 AI 이미지 사용
                    use_real_image = False
                    decision_method = "실제 이미지 없음"
                
                decision_elapsed_time = time.time() - decision_start_time
                logging.info(f"이미지 타입 결정 ({decision_method}): {decision_elapsed_time:.2f}초 소요")
                
                # 이미지 선택 및 매칭 시간 측정
                image_selection_start_time = time.time()
                if use_real_image and image_list and image_captions:
                    logging.info(f"섹션: '{combined_text[:50]}...'에 실제 이미지 매칭")
                    total_real_images += 1
                    
                    # GPT를 사용하여 섹션과 이미지 캡션 매칭
                    gpt_start_time = time.time()
                    url, header, request = setup_gpt_request(combined_text, image_captions, used_indices)
                    best_idx = execute_gpt(url, header, request)
                    gpt_elapsed_time = time.time() - gpt_start_time
                    
                    # 유효한 인덱스인지 확인
                    if 0 <= best_idx < len(image_paths):
                        # 이미 사용된 인덱스에 추가
                        used_indices.append(best_idx)
                        
                        # 매칭된 이미지의 원본 URL 찾기
                        best_path = image_paths[best_idx]
                        original_url = path_to_original_url.get(best_path, "")
                        
                        matched_images.append(original_url)
                        image_types.append("real")
                        
                        logging.info(f"  - GPT 매칭 성공: 인덱스 {best_idx}, 캡션: '{image_captions[best_idx]}'")
                        logging.info(f"  - GPT 매칭 소요 시간: {gpt_elapsed_time:.2f}초")
                    else:
                        # GPT 매칭 실패 시 AI 이미지로 대체
                        logging.warning(f"  - GPT 매칭 실패, AI 이미지로 대체")
                        ai_start_time = time.time()
                        ai_url = execute_flux(prompt)
                        ai_elapsed_time = time.time() - ai_start_time
                        logging.info(f"  - AI 이미지 생성: {ai_elapsed_time:.2f}초 소요")
                        
                        matched_images.append(ai_url)
                        image_types.append("ai")
                        total_real_images -= 1
                        total_ai_images += 1
                else:
                    logging.info(f"섹션: '{combined_text[:50]}...'에 AI 이미지 생성")
                    total_ai_images += 1
                    
                    ai_start_time = time.time()
                    ai_url = execute_flux(prompt)
                    ai_elapsed_time = time.time() - ai_start_time
                    logging.info(f"  - AI 이미지 생성: {ai_elapsed_time:.2f}초 소요")
                    
                    matched_images.append(ai_url)
                    image_types.append("ai")
                
                image_selection_elapsed_time = time.time() - image_selection_start_time
                logging.info(f"이미지 선택/생성 완료: {image_selection_elapsed_time:.2f}초 소요")
                
                section_elapsed_time = time.time() - section_start_time
                logging.info(f"섹션 그룹 {group_idx+1} 처리 완료: {section_elapsed_time:.2f}초 소요")
            
            # 결과
            result_item = {
                "category": category,
                "title": title,
                "section": sections,
                "image": matched_images,
                "image_types": image_types  # 내부 로깅을 위한 이미지 타입 정보 유지
            }
            results.append(result_item)
            
            category_elapsed_time = time.time() - category_start_time
            logging.info(f"카테고리 '{category}' 처리 완료: {len(matched_images)}개 이미지, {category_elapsed_time:.2f}초 소요")
            
    finally:
        # 메모리 정리
        if model:
            cleanup_blip(model, processor)
    
    matching_elapsed_time = time.time() - matching_start_time
    logging.info(f"이미지 매칭 처리 완료: {matching_elapsed_time:.2f}초 소요")
    
    # 통계 출력
    elapsed_time = time.time() - start_time
    logging.info(f"이미지 매칭 완료: 총 {total_sections}개 섹션, {elapsed_time:.2f}초 소요")
    if total_sections > 0:
        logging.info(f"- 실제 이미지: {total_real_images}개 ({100*total_real_images/total_sections:.1f}%)")
        logging.info(f"- AI 이미지: {total_ai_images}개 ({100*total_ai_images/total_sections:.1f}%)")
    
    logging.info(f"시간 분석:")
    logging.info(f"- 스크래핑: {scrap_elapsed_time:.2f}초 ({scrap_elapsed_time/elapsed_time*100:.1f}%)")
    logging.info(f"- 다운로드: {download_elapsed_time:.2f}초 ({download_elapsed_time/elapsed_time*100:.1f}%)")
    if blip_load_time > 0:
        logging.info(f"- BLIP 로드: {blip_load_time:.2f}초 ({blip_load_time/elapsed_time*100:.1f}%)")
    if caption_time > 0:
        logging.info(f"- 캡션 생성: {caption_time:.2f}초 ({caption_time/elapsed_time*100:.1f}%)")
    logging.info(f"- 이미지 매칭: {matching_elapsed_time:.2f}초 ({matching_elapsed_time/elapsed_time*100:.1f}%)")
    
    return results