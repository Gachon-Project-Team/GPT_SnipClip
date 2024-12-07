import json
import api_key
import requests


#이미지만 추출 
def extracted_img(news):
    result = []
    for category, articles in news.items():
        category_data = {
            "category": category,
            "image": [(article["image"], article["url"]) for article in articles]
        }
        result.append(category_data)
    return result

#category 기반 이미지 매핑
def merge_script_image(script, image):
    # 1. 이미지 데이터를 category 기준으로 매핑
    image_mapping = {item["category"]: item["image"] for item in image}
    section_mapping = {item["category"]: item["section"] for item in image}

    # 2. 스크립트 데이터를 순회하며 병합
    result = []
    for item in script:
        merged_item = {
            "category": item["category"],
            "title": item["title"],
            "section": section_mapping.get(item["category"], None),
            "image": image_mapping.get(item["category"], None) 
            }
        result.append(merged_item)

    return result

#ai 사용시 이미지 생성 호출
def gen_ai(script):
    prompt = generate_prompt(script)
    image = execute_dalle(prompt)
    with open('prompt.json', 'w', encoding='utf-8') as file:
        json.dump(prompt, file, ensure_ascii=False, indent=4)

    return merge_script_image(script, image)

#대본 기반 prompt생성 실행  
def generate_prompt(script):
    result = []
    for s in script:
        category=s["category"]
        script=s["content"]
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

# #4모델 쓸때만 사용 
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
                "Your task is to divide the script into logical sections and create detailed prompts for image generation using DALL·E. "
                "Each result must include the following components: "
                "- The category of the script "
                "- The script divided into sections "
                "- A descriptive image prompt for each section, suitable for DALL·E generation. "
                "Ensure that the prompts are vivid, detailed, and contextually aligned with the narrative. "
                "Additionally, include any other related queries or requirements from the user."
            )
        },
        {
            "role": "user",
            "content": (
                "Divide the following script into exactly 2 sections. For each section, create a descriptive prompt suitable for DALL·E image generation. "
                "Include the category, the script divided into sections, prompts for each section, and any other specified details. "
                "Return the result in the following format: "
                "{\"category\": \"category_name\", \"section\": [\"section 1 text\", \"section 2 text\"], \"prompt\": [\"prompt for section 1\", \"prompt for section 2\"]}. "
                f"Category: {category}. Script: {script}. Include any additional user requirements if applicable."
            )
        }
    ]
}


    return url, header, request

#dalle 실행
def execute_dalle(prompt):
    result=[]
    for p in prompt:
        category=p["category"]
        section=p["section"]
        prompt=p["prompt"]
        
        img_url=[]
        for s in prompt:
            innerList=[]
            url, header, data = set_up_dalle(s)
            response = requests.post(url, headers=header, json=data)
            if response.status_code == 200:
                image_urls = response.json()["data"][0]["url"]
                innerList.append(image_urls)
                img_url.append(innerList)
            else: 
                print(f"* execute_dalle * Error: {response.status_code}, {response.text}")
        dict = {
            "category":category,
            "section": section,
            "image": img_url
        }
        result.append(dict)
    
    return result

#dalle setup
def set_up_dalle(prompt):
    key = api_key.get_gpt_key()
    url = "https://api.openai.com/v1/images/generations"
    
    header = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    data = {
    "prompt": f"{prompt}",
    "n": 1,  # 생성할 이미지 개수
    "size": "1024x1024"  # 이미지 크기
    }

    return url, header, data

#메인 함수 
def generate_image(news, script, ai): 
    #카테고리 별 이미지랑 url만 추출
    image_url = extracted_img(news)

    #ai 사용 여부에 따라 result 생성
    #ai 사용하면 스크립트에 맞춘 그림 전부 생성 
    if (ai) : 
        return gen_ai(script)
    #ai 사용 안하면 실제 기사에 있던 이미지만 사용
    else: 
        return merge_script_image(script, image_url)
    
    
#test code
with open('scrap.json', 'r', encoding='utf-8') as file:
        news = json.load(file)
with open('scrip.json', 'r', encoding='utf-8') as file:
        script = json.load(file)

final = generate_image(news, script, 1)
with open('ai.json', 'w', encoding='utf-8') as file:
        json.dump(final, file, ensure_ascii=False, indent=4)