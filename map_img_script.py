import os
import json
import shutil
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPModel, CLIPProcessor
import logging
import numpy as np
from piq import ssim


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

def script_embedding(device, model, processor, query, image):
    result = []
    for category in image:
        embedding = []
        for section in category["section"]:
            try:
                text_inputs = processor(text=query + " " + section, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    text_outputs = model.get_text_features(**text_inputs)
                text_embedding = text_outputs.squeeze().to("cpu")
                embedding.append(text_embedding)
            except Exception as e:
                logging.error(f"* script_embedding * Failed processing section: {e}")
        result.append({
            "category": category["category"],
            "text_embedding": embedding
        })
    return result

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

    for img in image:
        try:
            url = img[0]  # URL 추출
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
                if is_similar_ssim(save_path, existing_images, threshold=0.9):
                    logging.info(f"Image {url} is too similar to existing images. Skipping.")
                    os.remove(save_path)  # 유사하면 저장된 파일 삭제
                    continue

            # 유사하지 않거나 첫 이미지인 경우 저장
            existing_images.append(save_path)
            successful_downloads.append(img)  # ["사진다운url", "출처뉴스url"] 형식 유지
            logging.info(f"Saved successfully: {save_path}")
            idx += 1
        
        except Exception as e:
            logging.error(f"* download_img * Failed to process {img[0]}: {e}")
    
    logging.info(f"Final successful downloads: {successful_downloads}")  # 디버깅: 최종 성공 리스트 출력
    return successful_downloads  # 성공적으로 다운로드된 이미지만 반환

def image_embedding(device, model, processor):
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


def cosine_similarity(script_val, image_val):
    result = []
    for script_category in script_val:    
        # 텍스트 임베딩 순회
        index=[]
        for section in script_category["text_embedding"]:
            section_similarity = []
            # 이미지 임베딩과의 개별 유사도 계산
            for img in image_val:
                similarity = torch.nn.functional.cosine_similarity(section, img, dim=0)
                section_similarity.append(similarity.item())  # 유사도 값을 리스트에 저장
            # 가장 높은 유사도를 가진 이미지 인덱스 찾기
            max_similarity_idx = section_similarity.index(max(section_similarity))
            # 중복 방지 처리
            while max_similarity_idx in index:
                section_similarity[max_similarity_idx] = float('-inf')  # 유사도를 임시로 제거
                max_similarity_idx = section_similarity.index(max(section_similarity))
            index.append(max_similarity_idx)  # 선택된 이미지 인덱스 추가
        # 해당 텍스트 카테고리의 결과 저장
        result.append({
            "category": script_category["category"],
            "map_index": index
        })
    print(result)
    return result

def gen_matching_file(image, map_index):
    result=[]
    for m in map_index:
        img_result=[]
        for i in m["map_index"]:
            img_result.append(image[i])
        result_dict = {
            "category":m["category"],
            "image":img_result
        }
        result.append(result_dict)
    
    return result

def mapping_image_script(query, image, script):
    device, model, processor = set_model()
    down_image = download_img(image)
    script_val = script_embedding(device, model, processor, query, script)
    image_val = image_embedding(device, model, processor)
    similarity = cosine_similarity(script_val, image_val)
    result = gen_matching_file(down_image, similarity)
    
    return result

# if __name__ == "__main__":
#     device, model, processor = set_model()
#     query="일본"
#     with open ('script.json', 'r', encoding='utf-8') as file:
#         script=json.load(file)
#     script_val = script_embedding(device, model, processor, query, script)
#     image_val = image_embedding(device, model, processor)
#     similarity = cosine_similarity(script_val, image_val)
#     result = gen_matching_file(image, similarity)

#     with open ('result.json', 'w', encoding='utf-8') as file:
#         json.dump(result, file,  ensure_ascii=False, indent=4)