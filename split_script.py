import requests
import json
import api_key
  
def execute_section_gpt(image):
    result = []
    for i in image:
        script=i["section"]
        url, header, request = setup_split_section(script)
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
 
def setup_split_section(script):
    key = api_key.get_gpt_key()
    url = "https://api.openai.com/v1/chat/completions"
    header = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    request = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a content assistant specializing in script summarization for short-form video creation. "
                    "Your task is to take an input script divided into sections and reformat it into exactly 10 concise sentences. "
                    "Each sentence must be logically split, preserve context, and be optimized for short-form video captions. "
                    "Ensure the sentences flow naturally and avoid abrupt or meaningless splits. "
                    "Return the result as a JSON array in this format: [\"sentence1\", \"sentence2\", ..., \"sentence10\"]. "
                    "Do not include any additional text or formatting outside the JSON array."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is the script input: {script}. "
                    "Combine the sections, then split the script into exactly 10 logically concise sentences. "
                    "Ensure the result is returned in Korean and preserves the original meaning of the script."
                    "Return the result as a JSON array."
                )
            }
        ]
    }
    return url, header, request


def merge_result(section, image):
    result=[]
    num = len(section)
    for i in range(num):
        image[i]["section"]=section[i]
        result.append(image[i])

    return result

def generate_result(image):
    section = execute_section_gpt(image)
    result = merge_result(section, image)

    return result

if __name__ == "__main__":
    with open ('result.json', 'r', encoding='utf-8') as file:
        image = json.load(file)
    result = generate_result(image)
    with open ('split.json', 'w', encoding='utf-8') as file:
        json.dump(result, file,  ensure_ascii=False, indent=4)
