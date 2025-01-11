import json
from openai import OpenAI
from IPython.display import Image, display
import api_key
import requests

#각 분류별로 스크립트 생성 (섹션 생성)

# 4모델 쓸 때 반환 결과 추가 처리에 사용 
def clean_gpt_response(raw_content):
    raw_content = raw_content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]  # ```json 제거
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]  # ``` 제거
    return raw_content

# 각 카테고리 별 내용 요약 실행 
def execute_script(news, query):
    result=[]
    for category, articles in news.items():
        url, header, request = setup_gpt_request(category, json.dumps(articles), query)
        gpt_result = execute_gpt(url, header, request) #반환결과 category, title, section, image 딕셔너리
        print(gpt_result)
        if gpt_result:
            image=[]
            for a in articles:
                image.append([a["image"], a["url"]])
            gpt_result["image"]=image
            result.append(gpt_result)
            
    return result

def setup_gpt_ai_request(sections):
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
                "Your task is to determine if a section of text requires a real image (0) or an AI-generated image (1). "
                "For each section, you must analyze whether the subject is a specific entity or object, or if the subject is more abstract or general. "
                "Use the following guidelines:\n\n"
                "1. If the subject is a specific object, entity, or character (e.g., a person, animal, well-known location, etc.), return 0 for a real image.\n"
                "2. If the subject is abstract or general, and doesn't refer to a specific object, return 1 for an AI-generated image.\n"
                "3. If both types of images are suitable, return both real and AI images in a mixed format.\n\n"
                "The result should be a list of 0s and 1s, where 0 represents a real image and 1 represents an AI-generated image. "
                "The result should be in the format of lists of lists. For example: [[0, 1], [0, 0, 1]]\n\n"
                "Each section corresponds to one sentence or unit of text, so you should provide one result per section."
            )
        },
        {
            "role": "user",
            "content": (
                f"Here are the sections to analyze: {sections}. Please determine the image type for each section based on the above criteria. "
                "Return format: ex. [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]...]. Do not return any other description. Return only the list."
            )
        }
    ]
}
    
    return url, header, request



# GPT 실행
def execute_gpt(url, header, request):
    response = requests.post(url, headers=header, json=request)
    
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        try:
            # JSON 파싱
            clean_content = clean_gpt_response(raw_content)
            categories = json.loads(clean_content)
        except json.JSONDecodeError as e:
            print("* gen_script/execute_gpt * Failed to parse JSON:", raw_content, e)
            categories = None
    else:
        print(f"* gen_script/execute_gpt * response Error: {response.status_code}")
        categories = None
        
    return categories
    
# GPT 사용 뉴스 요약
def setup_gpt_request(category, news, query): #키워드 쿼리, news가 catecory하나에대한 기사들 모음
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
                "Your task is to summarize a news story accurately and concisely, preserving the facts while highlighting the key points. "
                "The summary should be in a conversational tone, written in Korean, and designed to be easily understood when spoken aloud. "
                "Ensure the summary includes only the most important information, and avoid any form of exaggeration or distortion of facts. "
                "For each section, please ensure the summary is grounded in the actual facts and does not veer into speculation. "
                "Include specific, relevant details that provide a clear understanding of the story without unnecessary elaboration. "
                "The summary must be divided into 10 distinct sections, each one presenting a key idea in a concise and understandable way. "
                "For the references, please include **all the articles that cover the key points** of the summary. However, if the same content appears in multiple articles, "
                "please include only one of them in the references to avoid redundancy. "
                "If there are multiple articles that together provide a comprehensive understanding of the story, include all the unique articles in the references, but exclude duplicates."
            )
        },
        {
            "role": "user",
            "content": (
                f"Summarize the news articles related to the keyword '{query}' into a 50-second voice presentation script, ensuring accuracy and clarity. "
                "The summary should focus on the key facts of the story, maintaining an unbiased and factual tone. "
                "Please make sure to break the summary into exactly 10 sections, each clearly summarizing one point of the story. "
                "Each section must be based on factual information, and for each section, provide the reference URLs that support the information in that section. "
                "For the references, include **all the articles that cover the key points of the story**. However, if identical information appears in multiple articles, include only one of them to avoid repetition. "
                "If there are multiple articles that together cover all the important points, include all the unique articles in the reference list. "
                "The following format should be used for the output:\n\n"
                f"category should be equal to {category}. Do not edit.\n"
                "{\n  \"category\": \"{category}\",\n  \"title\": \"{title}\",\n  \"sections\": [\"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\"],\n  \"references\": [\"\", \"\", \"\"]\n}\n\n"
                f"Here is the data: {news}. Please ensure that the summary is accurate and factual, with clear reference URLs for each section. Avoid including duplicate articles."
            )
        }
    ]
}


    return url, header, request

# ai 이미지 or 실제 이미지 구분 
def execute_image_map(section):
    url, header, request = setup_gpt_ai_request(section)
    result = execute_gpt(url, header, request) #반환결과 category, title, section, image 딕셔너리

    return result


# 메인 함수
def generate_script(news, query):
    # 카테고리별로 대본 생성 GPT에 요청 [{"category"="", "title"="", "section"="", "ai"=[], "reference"=[], "image"=[[사진url, 출처url],[사진url, 출처url]]}] 형태
    script = execute_script(news, query) 
    print(script)
    result=[]
    for i in script:
        sections=[]
        for k in range(0, 10, 2): 
            section = i["sections"][k]+i["sections"][k+1]
            sections.append(section)
        result.append(sections)    
    print(result)
    ai = execute_image_map(result)
    print("\n ** ai ** \n")
    print(ai)
    for i in range(len(ai)):
        script[i]["ai"]=ai[i]

    return script
