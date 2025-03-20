import requests
import generate_scrap
import torch
import logging
import shutil
from PIL import Image
from io import BytesIO
import flux
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import api_key

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

# 이미지 다운로드
def download_img(image): # image는[[url, ref url]] 형태
    save_dir = "./image"
    
    # 기존 디렉토리 존재시 삭제 후 새로 생성
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    idx = 0
    successful_downloads = []  # 다운로드 성공한 이미지 저장 리스트
    path=[]
    for i in image:
        try:
            url = i[0]  # URL 추출
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
            img_path = save_path
            path.append(img_path)

            # 이미지 저장
            img_data.save(save_path)
            successful_downloads.append(i)
            idx=idx+1
        except Exception as e:
            logging.error(f"* download_img * Failed to process {i[0]}: {e}")
    
    logging.info(f"Final successful downloads: {successful_downloads}")  
    return successful_downloads, path  

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
            "image": image_mapping.get(item["category"], None) 
            }
        result.append(merged_item)

    return result

def image_to_text(path):
    # MPS 디바이스 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # img to text 모델 및 프로세서 로드
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    # 이미지 로드해서 리스트에 저장
    open_img = [Image.open(i).convert("RGB") for i in path]

    inputs = processor(images=open_img, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    out = model.generate(**inputs)
    captions = [processor.decode(c, skip_special_tokens=True) for c in out]

    return captions

# GPT 사용 뉴스 분류 작업 요청 준비 
def setup_gpt_request(caption, script, index): #섹션 한 개만 들어오고 caption 여러 개 .. 
    key=api_key.get_gpt_key()
    url="https://api.openai.com/v1/chat/completions"

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
                "당신에게는 영상 대본 중 한 섹션과 이미지를 캡션화한 데이터가 주어집니다"
                "당신의 역할은 대본과 가장 잘 매칭되는 이미지 캡션을 찾고 해당 캡션의 인덱스를 반환하는 것 입니다."
                f"단, 매칭된 이미지 캡션의 인덱스가 {index} 이 리스트에 있는 번호들과 중복된다면, 중복되지 않도록 재매칭 하세요."
                "또 아래와 같은 규칙을 지켜주세요"
                "1. 캡션의 인덱스는 0부터 시작합니다. index를 잘못 계산하지 않도록 주의하세요. "
                "2. 이미지 캡션에는 특정 개체에 대한 일반화된 표현이 사용될 수 있습니다. 예를 들어 '푸바오'는 '판다'로만 표현되거나, 소설작가 한강은 '치마를 입은 여성' 으로 표현됩니다. 이를 고려하세요."
                "3. 캡션은 이미지에 있는 글자를 잘못 인식해 표현했을 수 있습니다 이를 고려하세요."
                "4. '대본과 잘 매칭되는 이미지'란, 대본을 읽을 때 시각적 자료로 사용할 수 있을만한 즉, 대본을 이해하는데 도움을 줄만한 이미지를 의미합니다."
                "5. 실제 세계를 찍은 이미지가 아니라 포토샵으로 만든 것같은 이미지에 대한 캡션은 매칭을 자제해주세요. 이미지 캡션엔 기사 로고 이미지에 대한 캡션이 섞여있을 수 있으니 주의해주세요"
                "6. 캡션 인덱스 이외의 추가 설명은 반환하지 마세요. 인덱스 숫자만 반환하세요. "
                "7. a close up of a newspaper with a news paper on it 혹은 이와 유사한 캡션은 뉴스 로고에 대한 캡션이니 어떤 대본과도 매칭시키지 마세요."
            )
        },
        {
            "role": "user",
            "content": (
                f"Here is the news script: {script} "
                f"Here are the image captions: {caption} "
                f"{index}에 있는 번호와 중복된 번호를 반환하지 않도록 매칭을 진행하세요"
                "Output example: 3"
            )
        }
    ]
}

    return url, header, request

# 4모델 쓸 때 반환 결과 추가 처리에 사용 
def clean_gpt_response(raw_content):
    raw_content = raw_content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]  # ```json 제거
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]  # ``` 제거
    return raw_content

# GPT API 실행 및 결과 JSON 형태로 변환
def execute_gpt(url, header, request):
    # GPT API 호출
    response = requests.post(url, headers=header, json=request)
    # GPT 응답 처리
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        clean_content = clean_gpt_response(raw_content)
        result = int(clean_content)
        return result
    else:
        print(f"* generate_image/execute_gpt * response Error: {response.status_code}")
        return 0

# 한 카테고리별로 이미지 매칭 실행
def image_mat(caption, category, query, image_list):
    section = [category["sections"][i]+category["sections"][i+1] for i in range(0, 10, 2)] #섹션 5개로 (script만들 때 10개로 나뉘는데 사진은 5개 필요하니까)
    index_list=[]
    image = []
    for i in range(len(category["ai"])): #ai 사용 여부에 따라 이미지 매칭 방식이 clip을 사용하는지 flux를 사용하는지로 나뉨
        if category["ai"][i]==0: 
            #ai 안 쓸 때 
            #gpt통해 image to text로 변경, script와 이미지 설명 text 간 매칭
            url, head, request = setup_gpt_request(caption, section[i], index_list)
            print(index_list)
            index = execute_gpt(url,head,request)
            index_list.append(index)
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
    image_list, path = download_img(image) #다운된 이미지들 목록 [[url, reference]] 형식, path-> 이미지 로컬 path 경로
    caption = image_to_text(path) #이미지 캡션들만 저장
    image_result=[] 
    for category in script: 
        category_result = image_mat(caption, category, query, image_list) #한 카테고리에 대한 이미지 매칭 결과 {"category": "category_1, "image": [[url, ref]]} 형식
        image_result.append(category_result) #모든 카테고리에 대한 결과를 한 리스트로 묶음
    result = merge_script_image(script, image_result) #해당 이미지 매칭 파일을 카테고리별로 섹션, 타이틀 등과 매칭해서 최종 파일 생성

    return result


