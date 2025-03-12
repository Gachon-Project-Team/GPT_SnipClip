import json
import urllib.request
from newspaper import Article
import requests
import api_key

# 뉴스 스크랩 및 분류

# 뉴스 스크랩
def execute_scrap(query):
    client_id, client_secret = api_key.get_naver_key()
    encText = urllib.parse.quote(query) 
    url = f"https://openapi.naver.com/v1/search/news?query={encText}&sort=sim&display=100"
    data = [] #기사를 저장할 리스트, 딕셔너리 리스트 형태, key = id, title, content, image, url
    
    # naver 기사 api 링크 접속
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    
    #네이버 뉴스 접속 성공시 뉴스 스크래핑 (url만 가져옴)
    if rescode == 200:
        response_body = response.read()
        response_json = json.loads(response_body.decode("utf-8"))   
        links = [item.get("link", "") for item in response_json.get("items", [])]
        article_id = 0  # 각 뉴스 별 ID 생성
        for link in links:
            try:
                article_id += 1
                #link로 접속해 스크래핑
                article = Article(link, language="ko")
                article.download()
                article.parse()
                #긁어온 정보들 딕셔너리 형태로 저장
                article_dict = {
                    "id": article_id,
                    "title": article.title,
                    "content": article.text,
                    "image": article.top_image,
                    "url": link
                }
                # 내용이 없는 기사 제외하고 data에 저장 
                if article_dict["content"]:
                    data.append(article_dict)
            #접속 안되는 링크 console에 error 
            except Exception as e:
                print(f"* news_scrap * Failed to process article link - {link}: {e}")
    #네이버 뉴스 접속 실패시 console에 error출력 
    else:
        print("* news_scrap * Failed to process Naver News, Error Code:", rescode)

    return data

# GPT 사용 뉴스 분류 작업 요청 준비 
def setup_gpt_request(data, query):
    key=api_key.get_gpt_key()
    url="https://api.openai.com/v1/chat/completions"
    keys = ["id", "title"]
    filtered_data = [{key: item[key] for key in keys} for item in data]
    filtered_data_json = json.dumps(filtered_data, ensure_ascii=False)   
    
    header = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    request = {
        "model": "gpt-4o-mini",
        "messages": [
    {
        "role": "system",
        "content": "You are an advanced news classifier and content curator."
    },
    {
        "role": "user",
        "content": (
            f"The dataset contains 100 news article titles scraped using the query '{query}'. "
            "Your task is to group these titles into at least three distinct categories with minimal overlap. "
            "Ensure that each category represents a unique theme suitable for a short-form video. "
            "If two or more groups share highly similar topics, merge them into one category. "
            f"Exclude any titles that are irrelevant or only tangentially related to the {query} "
            "Return your answer strictly in JSON format with the following structure: {\"category_name1\": [news_id, ...], \"category_name2\": [news_id, ...]}"
            f"Here is the data: {filtered_data_json}"
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
def execute_gpt(data, url, header, request):
    # GPT API 호출
    response = requests.post(url, headers=header, json=request)
    result={}

    # GPT 응답 처리
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        clean_content = clean_gpt_response(raw_content)
        try:
            # JSON 파싱
            categories = json.loads(clean_content)
            # 각 카테고리별 데이터 저장 
            for category, ids in categories.items():  
                filtered_news = [article for article in data if article["id"] in ids]
                result[f"{category}"] = filtered_news
        except json.JSONDecodeError as e:
            print("* news_scrap/execute_gpt * Failed to parse JSON:", raw_content, e)
    else:
        print(f"* news_scrap/execute_gpt * response Error: {response.status_code}")
    
    return result
    
# 메인 함수
def news_scraper(query):
    #스크랩된 뉴스들 저장 
    data = execute_scrap(query)
    #gpt api connection 준비 
    url, header, request = setup_gpt_request(data, query)
    #gpt api와 커넥션 후 분류된 최종 결과 저장 
    result = execute_gpt(data, url, header, request)
    
    return result