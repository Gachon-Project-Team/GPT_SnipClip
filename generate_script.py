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
                "Your task is to determine whether an image should be a real image (0) or an AI-generated image (1) based on the given text section."
                "Consider that the image will be used in a card news format, so it must visually match the content of the text."
                "Follow these rules when deciding between a real image and an AI-generated image:"
                "1. If the subject refers to a specific, real-world entity that exists or has existed (e.g., a real person, historical event, well-known location, recognizable object, or natural phenomenon), return 0 (real image)."
                "2. If the subject is abstract, conceptual, futuristic, or does not refer to a well-defined real-world entity, return 1 (AI-generated image)."
                "3. If both a real and AI-generated image could work equally well, provide a mix of 0s and 1s based on what best suits the card news format."
                "The result should be a list of 0s and 1s, where:"
                "- 0 = real image"
                "- 1 = AI-generated image"
                "Ensure that your decision is based on which type of image will best represent the meaning and intent of the text in the card news format."
                "Examples:"
                "문장: 대학 측은 2023년 5월과 11월에 각각 입학설명회를 개최할 예정입니다. 신입생 모집을 위해 대학은 다양한 홍보 전략을 시행하고 있으며, SNS를 통한 정보 제공도 늘리고 있습니다."
                "Result: AI-generated image (1), because although 'the university' refers to a specific entity, the description of admission briefings and promotional strategies makes it difficult to find a suitable real image. Using an AI-generated image would be more appropriate."
                "문장: 가천대학교는 이러한 프로그램을 통해 창의적이고 혁신적인 인재를 양성하고자 합니다. 2025 GCSC는 다양한 산업 분야의 전문가들이 참여하여 멘토링을 제공합니다."
                "Result: Real image (0), because the text refers to 2025 GCSC held at Gachon University, and using an actual photo of the university better matches the context rather than an AI-generated image."
                "문장: 가천대학교는 글로벌 인재 양성을 목표로 다양한 해외 연수 프로그램을 운영하고 있습니다. 대학 측은 지속 가능한 발전을 위해 환경 친화적인 캠퍼스를 조성할 계획입니다."
                "Result: AI-generated image (1), because the text discusses future-oriented programs such as study abroad and eco-friendly campus development, making an AI-generated image more suitable."
                "문장: 앰브로스는 생명과학 분야의 선구자로 평가받고 있으며, 그의 연구는 여러 생명체의 유전자 조절 메커니즘을 이해하는 데 기여했다. 특히, 마이크로RNA는 유전자 발현 조절에 중요한 역할을 하며, 질병 연구에도 활용되고 있다."
                "Result: Real image (0), because Ambros is a specific individual, and using a real image of him is more appropriate for this context."
                "문장: 노벨 평화상 후보로는 환경 문제에 기여한 인사들이 거론되고 있습니다. 노벨 문학상은 특히 문학계의 새로운 목소리를 가진 작가에게 돌아갈 가능성이 높습니다."
                "Result: AI-generated image (1), because no specific individuals are mentioned, making it difficult to find a suitable real image. Therefore, an AI-generated image would be more appropriate."
                "문장: 전문가들은 이러한 결과가 과학계에 미치는 영향을 분석 중입니다. 향후 노벨상 제도의 공정성과 신뢰성을 회복하기 위한 논의가 필요할 것으로 보입니다."
                "Result: AI-generated image (1), because the content is abstract, making it difficult to find a suitable real photo. An AI-generated image would be more appropriate."
                "문장: 탄핵 관련 법적 절차와 기준에 대한 논의도 함께 이루어지고 있습니다. 국회 내에서의 협상과 여론의 변화가 탄핵 여부에 큰 영향을 미칠 것으로 보입니다."
                "Result: Real image (0), because the discussion about impeachment within the National Assembly makes an actual photo of the assembly more appropriate. Given the factual importance of the political content, an AI-generated image should not be used."
                "문장: 최근 탄핵 관련 논의가 활발해지고 있습니다. 정치권에서는 탄핵 절차에 대한 다양한 의견이 나오고 있습니다. 한 정치인은 탄핵이 필요하다고 주장하며, 정치적 책임을 강조했습니다."
                "Result: AI-generated image (1), because the phrase 'a politician' is vague, making it difficult to match with a real image. An AI-generated image is more appropriate."
                "문장: 여론조사 결과는 향후 정치적 결정에 큰 영향을 미칠 것으로 예상됩니다. 탄핵에 대한 논의는 국회에서도 활발히 이루어지고 있으며, 여러 정당이 입장을 내놓고 있습니다."
                "Result: Real image (0), because this text discusses factual political events. A real image of the National Assembly would be more appropriate."
                "문장: 탄핵 절차가 진행될 경우, 정치적 파장이 클 것으로 예상되고 있습니다. 일부 정치인은 탄핵이 정치적 목적을 위한 것이라고 비판하고 있습니다."
                "Result: AI-generated image (1), because 'some politicians' is a vague reference, making it difficult to match with a real image. An AI-generated image would be more suitable."
            )
        },
        {
            "role": "user",
            "content": (
                f"Here is a section to analyze: {sections}. Please determine the appropriate image type for each section based on the criteria above.\n"
                "Return format: [[0, 1, 0, 1, 0], [1, 0, 0, 1, 0] ...]; no other explanation, only the list."
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
                "Your task is to create a 50-second script of video news by summarizing the given articles."
                "The script should be divided into a total of 10 sections, each of which should be followed naturally when combined."
                "The script must be written in korean and All terminating endings must be in the formality. ex. -합니다. -습니다."
                "Ensure the script includes only the most important information and do not exaggeration or distortion of facts."
                "Do NOT add any information that is not in the articles and Do NOT mention the year or date in any section"
                "Keep the summary neutral and objective, avoiding personal opinions or interpretations."
                "Use specific names for people, animals, countries, and institutions as they appear in the articles."
                "It's 2025 now, please remember this moment and write the script"
            )
        },
        {
            "role": "user",
            "content": (
                f"Write the news script that related to the keyword '{query}'"
                "The summary should focus on the key facts of the story, maintaining an unbiased and factual tone. "
                "The following format should be used for the output:\n\n"
                f"category should be equal to {category}. Do not edit.\n"
                "{\n  \"category\": \"{category}\",\n  \"title\": \"{title}\",\n  \"sections\": [\"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\", \"\"]}"
                f"Here is the data: {news}"
            )
        }
    ], 
    "temperature": 0.4
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
    result=[]
    for i in script:
        sections=[]
        for k in range(0, 10, 2): 
            section = i["sections"][k]+i["sections"][k+1]
            sections.append(section)
        result.append(sections)    
    ai = execute_image_map(result)
    for i in range(len(ai)):
        script[i]["ai"]=ai[i]

    return script

