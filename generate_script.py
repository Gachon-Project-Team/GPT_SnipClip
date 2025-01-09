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
        if gpt_result:
            image=[]
            for a in articles:
                image.append([a["image"], a["url"]])
            gpt_result["image"]=image
            result.append(gpt_result)
            
    return result

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
                "Your task is to summarize a news story in a short script that can be delivered in under 50 seconds."
                "The script should be concise, within 150-200 words, and written in Korean with a conversational tone appropriate for natural speaking."
                "Focus on highlighting the key points clearly without making the summary too abstract."
                "The script should be written in a natural flow without thinking of the sectioning at first."
                "Once the script is completed, divide it into exactly 10 sections for a consistent flow, ensuring the natural continuation between sections."
                "For each pair of adjacent sections (e.g., sections 1-2, 2-3, 3-4, etc.), evaluate if AI image generation is appropriate based on the following criteria:"
                "- If the section contains a proper noun (e.g., names, places, titles) and it is a primary subject, set the value to 0 (real images are preferred)."
                "- If no such proper noun is included or the context allows AI generation, set the value to 1."
                "You will provide 5 values for the 'ai' field, based on the evaluation of the following pairs of sections: 1-2, 2-3, 3-4, 4-5, 5-6."
                "Ensure that the final AI image usage and real images are mixed appropriately based on context."
                "Include the source URL used to write the script for reference."
            )
        },
        {
            "role": "user",
            "content": (
                f"Summarize news articles related to the keyword '{query}' into a conversational, 50-second voice presentation script."
                "Focus on clarity and consistency in tone, ensuring the style is unified at the end of each sentence."
                "Provide the output in the following JSON format:\n\n"
                f"category must be same {category}. do not edit it"
                "{\n  \"category\": \"{category}\",\n  \"title\": \"{title}\",\n  \"sections\": [\"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\"],\n  \"ai\": [0, 0, 0, 0, 0],\n  \"references\": [\"\", \"\", \"\"]\n}\n\n"
                f"Here is the data: {news}"
            )
        }
    ]
}



    return url, header, request

# 메인 함수
def generate_script(news, query):
    # 카테고리별로 대본 생성 GPT에 요청 [{"category"="", "title"="", "section"="", "ai"=[], "reference"=[], "image"=[[사진url, 출처url],[사진url, 출처url]]}] 형태
    return execute_script(news, query) 