import json
import api_key
import requests
import generate_scrap
import torch
from diffusers import FluxPipeline

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
            "section": item["section"],
            "image": image_mapping.get(item["category"], None) 
            }
        result.append(merged_item)

    return result

#대본 기반 prompt생성 실행  
def generate_prompt(script):
    result = []
    for s in script:
        category=s["category"]
        script=s["section"]
        url, header, request = setup_prompt_gpt_request(category, script)
        gpt_result = execute_gpt(url, header, request)
        
        result.append(gpt_result)
        
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
def setup_prompt_gpt_request(category, script):
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
                "You are a content assistant specializing in analyzing scripts for short-form video creation. "
                "Your task is create detailed prompts for image generation using DALL·E. "
                "Each result must include the following components: "
                "- The category of the script "
                "- A single descriptive image prompt for each section, suitable for DALL·E generation. "
                "Ensure that the prompts are vivid, detailed, and contextually aligned with the narrative. "
                "Additionally, include any other related queries or requirements from the user."
            )
        },
        {
            "role": "user",
            "content": (
                "Each section is separated by quotation marks."
                "For each section, create a single descriptive image prompt suitable for DALL·E generation. "
                f"category value must be same \"{category}\". don't edit!!!!!! "
                "Return the result in the following format: "
                f"{{\"category\": {category}, \"prompt\": [\"prompt for section 1\", \"prompt for section 2\"...]}}"
                f"here is script {script}"
            )
        }
    ]
}

    return url, header, request

#title로 이미지 재검색
def scrap_image(news_image, script):
    image=[]
    for item in script:
        query = item["title"] 
        news = generate_scrap.news_scrap(query)
        inner_image=[]
        
        for existing_item in news_image:
            if existing_item["category"] == item["category"]:
                for i in existing_item["image"]:
                    inner_image.append(i)
        
        for n in news:
            inner_image.append([n["image"], n["url"]])
        inner_dict = {
            "category": item["category"],
            "image": inner_image,
            "section": item["section"]
        }
        image.append(inner_dict)
        
    return image

#이미지 전처리 
def preprocess_image(image):
    return 0

def extract_img(scrap):
    result = []
    for category, articles in scrap.items():
        img_result = []
        for item in articles:
            img_result.append([item["image"], item["url"]])
        result_dict = {
            "category": category,
            "image": img_result
        }
        result.append(result_dict)

    return result

# FLUX.1 파이프라인 설정 함수
def set_up_flux():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

# FLUX.1 이미지 생성 함수
def execute_flux(prompts):
    pipe = set_up_flux()

    result = []
    for prompt_data in prompts:
        category = prompt_data["category"]
        sections = prompt_data["section"]
        prompt_list = prompt_data["prompt"]

        img_urls = []
        for prompt in prompt_list:
            try:
                image = pipe(
                    prompt,
                    height=512,
                    width=512,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]

                filename = f"{category}_{hash(prompt)}.png"
                image.save(filename)
                img_urls.append(filename)

            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {e}")

        torch.cuda.empty_cache()

        result.append({
            "category": category,
            "image": img_urls
        })

    return result

#메인 함수 
def generate_image(news, script, ai): 
    news_image = extract_img(news)
    #카테고리 별 title 재검색, 이미지 전처리(중복 내용 제거, 적절한 5개 이미지만 뽑음)
    image = scrap_image(news_image,script) #additional_image.json 으로 확인! 
    with open('pre_image.json', 'w', encoding='utf-8') as file:
            json.dump(image, file, ensure_ascii=False, indent=4)
    pre_image = preprocess_image(image)
    
    #ai 사용 여부에 따라 result 생성
    #ai 사용하면 스크립트에 따라 프롬포트 작성, 생성
    if (ai) : 
        prompt = generate_prompt(script)
        ai_image = execute_flux(prompt)
        
        return merge_script_image(script, ai_image)
    #ai 사용 안하면 실제 기사에 있던 이미지만 사용
    else: 
        return merge_script_image(script, pre_image)

# #test code 
# with open('scrap.json', 'r', encoding='utf-8') as file:
#         scrap = json.load(file)
# with open('script.json', 'r', encoding='utf-8') as file:
#         script = json.load(file)
# eximage=extract_img(scrap)
# image=scrap_image(eximage, script)
# with open('additional_image.json', 'w', encoding='utf-8') as file:
#         json.dump(image, file, ensure_ascii=False, indent=4)