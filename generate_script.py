import json
from openai import OpenAI
from IPython.display import Image, display
import api_key
import requests

# 각 카테고리 별 내용 요약 실행 
def generate_script(news, query):
    result=[]
    for category, articles in news.items():
        url, header, request = setup_gpt_request(category, json.dumps(articles), query)
        gpt_result = execute_gpt(url, header, request) #반환결과 category, title, contents 딕셔너리
        if gpt_result:
            result.append(gpt_result)
        else: #gpt_result가 none 인 경우
            continue
        
    return result

# GPT 실행
def execute_gpt(url, header, request):
    response = requests.post(url, headers=header, json=request)
    
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        try:
            # JSON 파싱
            categories = json.loads(raw_content)
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
    # "model": "gpt-4o-mini",
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "system",
            "content": (
                "Your job is to summarize news articles into a concise short-form script designed to be spoken within 1 minute and 30 seconds. "
                "The script should be approximately 180 to 225 words long and written in Korean. "
                "Ensure the output is in the format: {\"category\": \"\", \"title\": \"\", \"content\": \"\"}. "
                f"The value of category must be '{category}'."
            )
        },
        {
            "role": "user",
            "temperature": 1.4,
            "content": (
                f"The following articles are related to the keyword '{query}'. "
                f"Read all the news and summarize it into a one-and-a-half-minute presentation script. approximately 200 words, "
                f"and generate an appropriate title in Korean. "
                "Ensure the tone is consistent throughout, and use a unified style for sentence endings."
                f"Return the response in this JSON format: {{\"category\": \"{category}\", \"title\": \"\", \"content\": \"\"}}.\n\n"
                f"{news}"
                )
            }
        ]
    } 

    return url, header, request

# 메인 함수
def generate_script(news, query):
    #카테고리별로 대본 생성 GPT에 요청 [{"category"="", "title"="", "content"=""}] 형태
    return generate_script(news, query) 