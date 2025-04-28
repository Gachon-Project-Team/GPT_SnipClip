import json
import api_key
import requests

# GPT 응답 후 코드 블록 제거
def clean_gpt_response(raw_content):
    raw_content = raw_content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    return raw_content

# GPT 응답 요청 실행
def execute_gpt(url, header, request):
    response = requests.post(url, headers=header, json=request)

    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        try:
            clean_content = clean_gpt_response(raw_content)
            return json.loads(clean_content)
        except json.JSONDecodeError as e:
            print("* generate_script/execute_gpt * Failed to parse JSON:", raw_content, e)
            return None
    else:
        print(f"* generate_script/execute_gpt * response Error: {response.status_code}")
        return None

# GPT 요청 구성
def setup_gpt_request(category, news, query):
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
                    "You are an assistant that summarizes articles into a Korean news script for an approximately 1 minute long video.\n"
                    "You must output exactly 10 Korean sentences in formal tone (-합니다, -습니다).\n"
                    "Only use information from the articles. Do not add or fabricate anything.\n"
                    "Do not mention any specific year or date. Use real names from the text.\n"
                    "Respond only in valid JSON format. Do not wrap with triple backticks or add explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Write a news script related to the keyword '{query}'.\n"
                    f"The category must be exactly: {category} (do not change).\n"
                    "Return JSON in the following format ONLY:\n"
                    "{\n"
                    f'  "category": "{category}",\n'
                    '  "title": "뉴스 요약 제목",\n'
                    '  "sections": ["문장1", "문장2", "문장3", "문장4", "문장5", "문장6", "문장7", "문장8", "문장9", "문장10"]\n'
                    "}\n\n"
                    f"Here is the article data:\n{news}"
                )
            }
        ],
        "temperature": 0.4
    }

    return url, header, request

# 카테고리별 GPT 요청 실행 및 결과 처리
def execute_script(news, query, use_real_image=True):
    result = []
    for category, articles in news.items():
        url, header, request = setup_gpt_request(category, json.dumps(articles), query)
        gpt_result = execute_gpt(url, header, request)

        if gpt_result:
            if use_real_image:
                # 스크랩 이미지 URL 정보도 넣을 수 있음
                image = []
                for a in articles:
                    image.append([a.get("image", ""), a.get("url", "")])
                gpt_result["image"] = image

            result.append(gpt_result)

    return result

# 메인 함수: 전체 스크립트 반환
def generate_script(news, query, use_real_image=True):
    script = execute_script(news, query, use_real_image)
    return script
