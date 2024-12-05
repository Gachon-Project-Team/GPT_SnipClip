import json
import urllib.request
from newspaper import Article
import requests
import api

#뉴스 스크랩
def news_scrap(query):
  client_id, client_secret = api.get_naver_key()
  encText = urllib.parse.quote(query) 
  url = f"https://openapi.naver.com/v1/search/news?query={encText}&display=100"
  data = [] #기사를 저장할 리스트, 딕셔너리 리스트 형태, key = id, title, content, image, url
  
  # 링크 접속
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
            print(f"*** Failed to process {link}: {e}")
  #네이버 뉴스 접속 실패시 console에 error출력 
  else:
    print("*** Failed to process Naver News, Error Code:", rescode)

  return data

#GPT 사용 뉴스 분류 작업 요청 준비 
def setup_gpt_request(data, query):
  key=api.get_gpt_key()
  url="https://api.openai.com/v1/chat/completions"
  keys = ["id", "title"]
  filtered_data = [{key: item[key] for key in keys} for item in data]
  filtered_data_json = json.dumps(filtered_data, ensure_ascii=False) 
  
  header = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
  }

  request = {
      "model": "gpt-3.5-turbo",
      "messages": [
          {
              "role": "system",
              "content": "You are an advanced news classifier and content curator."
          },
          {
              "role": "user",
              "content": (
                  f"The data contains news articles scraped using the query '{query}'. "
                  "Your task is to categorize the articles into coherent groups based on their topics, themes, or overall context, ensuring that each group could represent a single short-form video. "
                  "Strictly exclude any articles that are irrelevant to the query or unrelated to the main keywords. "
                  "Prioritize grouping articles with highly similar topics, key figures, events, or narratives together, avoiding broad or loosely connected categorizations. "
                  "Focus on maintaining the strongest thematic coherence within each group. "
                  "Return the result in strict JSON format, following this structure: {\"category_num (ex:category_1)\": [news_id, ...]} "
                  "Do not include any articles that are only tangentially related to the query or contain overly generalized or unrelated content. "
                  "Ensure that the classification is fine-grained and avoids overly broad categories. "
                  f"Here is the data: {filtered_data_json}"
              )
          }
      ]
  }
  
  return url, header, request

#GPT API 실행 및 결과 JSON 형태로 변환
def execute_gpt(data, url, header, request):
  # GPT API 호출
  response = requests.post(url, headers=header, json=request)
  result={}

  # GPT 응답 처리
  if response.status_code == 200:
      raw_content = response.json()["choices"][0]["message"]["content"]
      try:
          # JSON 파싱
          categories = json.loads(raw_content)
          # 각 카테고리별 데이터를 동적 변수로 생성
          for category, ids in categories.items():  
              filtered_news = [article for article in data if article["id"] in ids]
              #result[f"{category}"] = json.dumps(filtered_news)
              result[f"{category}"] = filtered_news
      except json.JSONDecodeError as e:
          print("Failed to parse JSON:", raw_content, e)
  else:
      print(f"response Error: {response.status_code}")
  
  return result
  
#**메인함수** 
def news_scraper(query):
  #스크랩된 뉴스들 저장 
  data = news_scrap(query)
  #gpt api connection 준비 
  url, header, request = setup_gpt_request(data, query)
  #gpt api와 커넥션 후 분류된 최종 결과 저장 
  result = execute_gpt(data, url, header, request)
  
  #result 콘솔에 프린트하면 짤려서 그냥 임시로 ... 해놓은거임... 지워도됨 
  with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
  print("Result saved to result.json")
  
  return result

