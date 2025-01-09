import os
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import logging

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

def script_embedding(device, model, processor, query, section):
    try:
        text_inputs = processor(text=query + " " + section, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
            text_embedding = text_outputs.squeeze().to("cpu")
    except Exception as e:
        logging.error(f"* script_embedding * Failed processing section: {e}")
    return text_embedding

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

def cosine_similarity(script_val, image_val): #script는 섹션 하나, image는 전체 이미지 임베딩한값임
    section_similarity=[]
    # 이미지 임베딩과의 개별 유사도 계산
    for img in image_val:
        similarity = torch.nn.functional.cosine_similarity(script_val, img, dim=0)
        section_similarity.append(similarity.item())  # 유사도 값을 리스트에 저장
        # 가장 높은 유사도를 가진 이미지 인덱스 찾기
    max_similarity_idx = section_similarity.index(max(section_similarity))
    #해당 인덱스 반환
    return max_similarity_idx
            
def mapping_image_script(section, query, image_list, image_val):
    device, model, processor = set_model()
    script_val = script_embedding(device, model, processor, query, section)
    index = cosine_similarity(script_val, image_val)
    result = image_list[index]
    
    return result
