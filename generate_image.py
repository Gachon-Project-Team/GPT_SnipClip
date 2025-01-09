import json
import api_key
import requests
import generate_scrap
import torch
from diffusers import FluxPipeline
import json
import os
import logging
import shutil
from PIL import Image
from piq import ssim
import numpy as np
from io import BytesIO
import map_img_script

#category 기반 이미지 매핑
def merge_script_image(script, image):
    # 1. category="", image=[] 형식으로 변경
    image_mapping = {item["category"]: item["image"] for item in image}
    
    # 2. 스크립트 데이터를 순회하며 병합
    result = []
    for item in script:
        merged_item = {
            "category": item["category"],
            "title": item["title"],
            "section": item["sections"],
            "reference": item["references"],
            "image": image_mapping.get(item["category"], None) 
            }
        result.append(merged_item)

    return result

#GPT 실행
def execute_gpt(url, header, request):
    response = requests.post(url, headers=header, json=request)
    
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        clean_content = clean_gpt_response(raw_content)
        try:
            # JSON 파싱
            categories = json.loads(clean_content)
        except json.JSONDecodeError as e:
            print("* gen_image/execute_gpt * Failed to parse JSON:", raw_content, e)
            categories = None
    else:
        print(f"* gen_image/execute_gpt * response Error: {response.status_code}")
        categories = None
        
    return categories

#4모델 쓸 때 반환 결과 추가 처리에 사용 
def clean_gpt_response(raw_content):
    raw_content = raw_content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]  # ```json 제거
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]  # ``` 제거
    return raw_content

#prompt 생성 gpt 쿼리 
def setup_prompt_gpt_request(section):
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
            "content": "You are a content assistant who specializes in script analysis for short-form video production. Your mission is to use image creation AI to create detailed prompts for image creation. When given a script section, generate a concise image prompt that accurately represents the content of that section. Do not include unnecessary words."
        },
        {
            "role": "user",
            "content": f"Here's the script section: {section}"
        }
    ]
}

    return url, header, request

#대본 기반 prompt생성 실행  
def generate_prompt(section):
    url, header, request = setup_prompt_gpt_request(section)
    gpt_result = execute_gpt(url, header, request)

    return gpt_result

#title로 이미지 재검색, 전체 이미지 [[url, reference], ] 형태
def scrap_image(script):
    result = []
    for item in script:
        query = item["title"] 
        news = generate_scrap.execute_scrap(query)
        result = result + item["image"]
        for n in news:
            result.append([n["image"], n["url"]])
            
    return result

# FLUX.1 파이프라인 설정 함수
def set_up_flux():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

# FLUX.1 이미지 생성 함수
def execute_flux(prompt):
    pipe = set_up_flux()

    try:
        # 생성할 이미지 파일 경로
        folder_path = "generate_image"
        os.makedirs(folder_path, exist_ok=True)  # 폴더가 없으면 생성
        
        # 이미지를 생성
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # 파일명 생성
        filename = f"{hash(prompt)}.png"
        image_path = os.path.join(folder_path, filename)  # 폴더 경로와 파일명 결합
        image.save(image_path)

        # URL 반환 (로컬 파일 경로)
        image_url = f"file://{image_path}"

    except Exception as e:
        print(f"* generate_image / execute_Flux * Error generating image for prompt '{prompt}': {e}")
        image_url = None

    torch.cuda.empty_cache()

    return image_url

def is_similar_ssim(image_path, existing_images, threshold=0.9):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 새 이미지 처리
        new_image = Image.open(image_path).resize((400, 400)).convert("RGB")
        new_image = np.array(new_image) / 255.0  # [0, 1] 정규화
        new_image = torch.tensor(new_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # 기존 이미지와 비교
        for existing_path in existing_images:
            existing_image = Image.open(existing_path).resize((400, 400)).convert("RGB")
            existing_image = np.array(existing_image) / 255.0
            existing_image = torch.tensor(existing_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            # SSIM 계산 (piq 사용)
            ssim_score = ssim(new_image, existing_image, data_range=1.0).item()

            if ssim_score >= threshold:
                return True  # 유사 이미지 찾음

    except Exception as e:
        print(f"Error during SSIM comparison: {e}")
        return False

    return False  # 유사 이미지 없음

def download_img(image):
    save_dir = "./image"
    
    # 기존 디렉토리 삭제 후 새로 생성
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    idx = 0
    successful_downloads = []  # 다운로드 성공한 이미지 저장 리스트
    existing_images = []  # 기존 저장된 이미지 경로 리스트

    for i in image:
        try:
            url = i[0]  # URL 추출
            logging.info(f"Processing image: {url}")
            response = requests.get(url)  # URL에서 이미지 가져오기
            response.raise_for_status()  # HTTP 상태 확인
            img_data = Image.open(BytesIO(response.content)).convert("RGB")
            file_extension = img_data.format.lower() if img_data.format else "jpg"

            supported_formats = ['jpeg', 'jpg', 'png', 'bmp', 'tiff']

            # 지원되지 않는 파일 형식 검사
            if file_extension not in supported_formats:
                raise ValueError(f"Unsupported image format: {file_extension}")

            # 이미지 크기 확인
            width, height = img_data.size
            if width <= 300 or height <= 300:  # 가로 또는 세로가 300 이하인 경우
                logging.info(f"Image {url} is too small (width: {width}, height: {height}). Skipping.")
                continue
            
            # 파일 저장 경로 생성
            file_name = f"image_{idx}.{file_extension}"
            save_path = os.path.join(save_dir, file_name)

            # 이미지 임시 저장
            img_data.save(save_path)

            # SSIM 비교: 기존 이미지와 유사한지 확인
            if existing_images:  # 기존 이미지가 있는 경우에만 비교
                if is_similar_ssim(save_path, existing_images, threshold=0.8):
                    logging.info(f"Image {url} is too similar to existing images. Skipping.")
                    os.remove(save_path)  # 유사하면 저장된 파일 삭제
                    continue

            # 유사하지 않거나 첫 이미지인 경우 저장
            existing_images.append(save_path)
            successful_downloads.append(i)  # ["사진다운url", "출처뉴스url"] 형식 유지
            logging.info(f"Saved successfully: {save_path}")
            idx += 1
        
        except Exception as e:
            logging.error(f"* download_img * Failed to process {i[0]}: {e}")
    
    logging.info(f"Final successful downloads: {successful_downloads}")  # 디버깅: 최종 성공 리스트 출력
    return successful_downloads  # 성공적으로 다운로드된 이미지만 반환

def mapping_image(category, query, image_list, image_val):
    section = [category["sections"][i]+category["sections"][i+1] for i in range(0, 10, 2)] 
    image = []
    for i in range(len(category["ai"])):
        if category["ai"][i]==0:
            #ai 안 쓸 때 
            image.append(map_img_script.mapping_image_script(section[i], query,  image_list, image_val))
        elif category["ai"][i]==1:
            #ai 쓸 때 
            prompt=generate_prompt(section[i])
            url = execute_flux(prompt)
            image.append([url, 0])
    result = {
        "category":category["category"],
        "image":image
    }
    return result 

#메인 함수 
def generate_image(script, query): 
    image = scrap_image(script) #이미지 추가 스크랩
    image_list = download_img(image) #다운된 이미지들 목록 [[url, reference 형식]]
    print(image_list)
    image_val = map_img_script.image_embedding() #image 폴더에 있는 것들 임베딩 리스트 형태
    image_result=[]    
    for category in script: 
        category_result = mapping_image(category, query, image_list, image_val)
        image_result.append(category_result)

    result = merge_script_image(script, image_result)

    return result