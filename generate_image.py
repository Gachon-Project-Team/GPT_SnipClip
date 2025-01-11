import requests
import generate_scrap
import torch
import os
import logging
import shutil
from PIL import Image
from piq import ssim
import numpy as np
from io import BytesIO
import flux
import os
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# flux 이미지 저장 폴더
IMAGE_SAVE_DIR = "generated_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# 이미지 추가 스크랩 및 전체 이미지를 하나의 리스트에 넣음 
def scrap_image(script, query):
    result = []
    news = generate_scrap.execute_scrap(query)
    for n in news:
            result.append([n["image"], n["url"]])   
    for item in script:
        result = result + item["image"]

    return result

# ssim 유사도 
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

# ssim 이용 유사도 체크 후 이미지 다운로드
def download_img(image): # image는[[url, ref url]] 형태
    save_dir = "./image"
    
    # 기존 디렉토리 존재시 삭제 후 새로 생성
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
            if width <= 300 or height <= 300:  # 가로 또는 세로가 300 이하인 경우 제거
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

# koClip 모델 설정
def set_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA")
    else:
        device = torch.device("cpu")
        logging.warning("Using CPU")

    model_name = "Bingsu/clip-vit-large-patch14-ko"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    return device, model, processor

# koClip 사용 대본 임베딩
def script_embedding(query, section):
    device, model, processor = set_model()
    try:
        text_inputs = processor(text=query + " " + section, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
            text_embedding = text_outputs.squeeze().to("cpu")
    except Exception as e:
        logging.error(f"* script_embedding * Failed processing section: {e}")
    return text_embedding

# koClip 사용 이미지 임베딩
def image_embedding():
    device, model, processor = set_model()
    result = []
    image_dir = "./image"

    # 이미지 폴더 순회
    for image_file in sorted(os.listdir(image_dir)):
        # 이미지 파일만 처리
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        image_path = os.path.join(image_dir, image_file)
        try:
            # 이미지 열기 및 임베딩 생성
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_outputs = model.get_image_features(**inputs)
            image_embedding = image_outputs.squeeze().to("cpu")  # 임베딩 벡터
            result.append(image_embedding)  # 결과 리스트에 추가
        except Exception as e:
            logging.error(f"* image_embedding * Failed processing {image_file}: {e}")
            result.append(torch.zeros(512))  # 오류 시 기본값 (512차원 영벡터)

    return result

# 대본과 이미지 임베딩 값 사이 코사인 유사도 확인
def cosine_similarity(script_val, image_val): #script는 섹션 하나 임베딩 값, , image는 전체 이미지 임베딩한값임
    section_similarity=[]
    # 이미지 임베딩과의 개별 유사도 계산
    for img in image_val:
        similarity = torch.nn.functional.cosine_similarity(script_val, img, dim=0)
        section_similarity.append(similarity.item())  # 유사도 값을 리스트에 저장
        # 가장 높은 유사도를 가진 이미지 인덱스 찾기
    max_similarity_idx = section_similarity.index(max(section_similarity))
    #해당 인덱스 반환
    return max_similarity_idx

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

# 한 카테고리별로 이미지 매칭 실행
def image_mat(category, query, image_list, image_val):
    section = [category["sections"][i]+category["sections"][i+1] for i in range(0, 10, 2)] #섹션 5개로 (script만들 때 10개로 나뉘는데 사진은 5개 필요하니까)
    image = []
    for i in range(len(category["ai"])): #ai 사용 여부에 따라 이미지 매칭 방식이 clip을 사용하는지 flux를 사용하는지로 나뉨
        if category["ai"][i]==0: 
            #ai 안 쓸 때 
             #koClip모델 설정 
            script_val = script_embedding(query, section[i]) #섹션 하나 임베딩
            index = cosine_similarity(script_val, image_val) #해당 섹션과 전체 이미지 값 코사인 유사도 검사, 가장 유사도 높은 이미지의 인덱스를 가져옴
            result = image_list[index] # 해당 이미지의 [url, ref url]을 result에 저장
            image.append(result) # 최종으로 반환할 이미지 리스트에 append
        elif category["ai"][i]==1:
            #ai 쓸 때 
            prompt=flux.generate_prompt(section[i]) #section i에 대한 프롬프트 생성 
            url = flux.execute_flux(prompt) #해당 프롬프트로 이미지 생성 
            image.append([url, 0]) #결과 url, ref는 없으니까 0 저장 
    result = {
        "category":category["category"],
        "image":image
    }
    return result 

#메인 함수 
def generate_image(script, query): 
    image = scrap_image(script, query) #이미지 추가 스크랩 및 전체 이미지 하나의 리스트로 생성
    image_list = download_img(image) #다운된 이미지들 목록 [[url, reference]] 형식
    image_val = image_embedding() #이미지 전체 임베딩 값 리스트 
    image_result=[] 
    for category in script: 
        category_result = image_mat(category, query, image_list, image_val) #한 카테고리에 대한 이미지 매칭 결과 {"category": "category_1, "image": [[url, ref]]} 형식
        image_result.append(category_result) #모든 카테고리에 대한 결과를 한 리스트로 묶음

    result = merge_script_image(script, image_result) #해당 이미지 매칭 파일을 카테고리별로 섹션, 타이틀 등과 매칭해서 최종 파일 생성

    return result