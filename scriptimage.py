import json
from openai import OpenAI
from IPython.display import Image, display
import api
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
            #아티클 하나 줄여서 보내거나 보내는 기사 수 제한 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            #esult.append(gpt_result)
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
            print("Failed to parse JSON:", raw_content, e)
            categories = None
    else:
        print(f"response Error: {response.status_code}")
        categories = None
        
    return categories
    
# GPT 사용 뉴스 요약
def setup_gpt_request(category, news, query): #키워드 쿼리, news가 catecory하나에대한 기사들 모음
    key=api.get_gpt_key()
    url="https://api.openai.com/v1/chat/completions"
    
    header = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    request = {
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

#데이터에서 이미지, 링크만 추출 
def extracted_img(news):
    result = []
    for category, articles in news.items():
        category_data = {
            category: [
                {"url": article["url"], "image": article["image"]}
                for article in articles
            ]
        }
        result.append(category_data)
    
    return result

# 메인 함수
# 대본/이미지 매칭 파일 생성 함수 
def generate_script_img(news, query, ai):
    #이미지, 출처 매칭 딕셔너리, 딕셔너리 리스트 형태고 딕셔너리 안엔 category: [{"url"="", "image"=""}] 형태 
    image = extracted_img(news)
    print(image)
    
    #카테고리별로 대본 생성 GPT에 요청 [{"category"="", "title"="", "content"=""}] 형태
    script = generate_script(news, query) 
    #test code
    # with open("scrip.json", "w", encoding="utf-8") as file:
    #     json.dump(script, file, ensure_ascii=False, indent=4)
    # print(f"Script saved")
    
    #생성된 대본에 맞춘 이미지
    # 이부분 추가 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
    # switch(ai) {
    #     #ai 이미지 사용x
    #     case 0: { 
    #         break;
    #         }
    #     #ai 이미지 사용o
    #     case 1: {
    #         break;
    #     }
    #}
    
    #return result


#test code
with open("result.json", "r", encoding="utf-8") as file:
    news = json.load(file) 
generate_script_img(news, "가천대학교", 3)