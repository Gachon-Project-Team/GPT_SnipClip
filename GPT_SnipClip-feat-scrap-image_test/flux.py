import os
import torch
from diffusers import FluxPipeline
import api_key
import json
import requests

# FLUX.1 파이프라인 설정 함수
def set_up_flux():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

# FLUX.1 이미지 생성 함수
def execute_flux(prompt):
    pipe = set_up_flux()

    try:
        # 생성할 이미지 파일 경로
        folder_path = "generate_image"
        os.makedirs(folder_path, exist_ok=True)  # 폴더가 없으면 생성
        
        # 이미지를 생성
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # 파일명 생성
        filename = f"{hash(prompt)}.png"
        image_path = os.path.join(folder_path, filename)  # 폴더 경로와 파일명 결합
        image.save(image_path)

        # URL 반환 (로컬 파일 경로)
        image_url = f"file://{image_path}"

    except Exception as e:
        print(f"* generate_image / execute_Flux * Error generating image for prompt '{prompt}': {e}")
        image_url = None

    torch.cuda.empty_cache()

    return image_url

#4모델 쓸 때 반환 결과 추가 처리에 사용 
def clean_gpt_response(raw_content):
    raw_content = raw_content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]  # ```json 제거
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]  # ``` 제거
    return raw_content

#prompt 생성 gpt 쿼리 
def setup_prompt_gpt_request(section):
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
            "content": ("You are a content assistant who specializes in script analysis for short-form video production."
                        "Your mission is to use image creation AI to create detailed prompts for image creation." 
                        "When given a script section, generate a concise image prompt that accurately represents the content of that section."
                        "Do not include unnecessary words.")
        },
        {
            "role": "user",
            "content": (
                "do not return any description except prompt(result)"
                f"Here's the script section: {section}"
                )

        }
    ]
}
    return url, header, request

#gpt 실행
def execute_gpt(url, header, request):
    response = requests.post(url, headers=header, json=request)
    
    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"]
        clean_content = clean_gpt_response(raw_content)
        categories = clean_content  # 텍스트 응답이므로 그냥 그대로 사용
    else:
        print(f"* flux /execute_gpt * response Error: {response.status_code}")
        categories = None
        
    return categories

#대본 기반 prompt생성 실행  
def generate_prompt(section):
    url, header, request = setup_prompt_gpt_request(section)
    gpt_result = execute_gpt(url, header, request)

    return gpt_result

#test
if __name__ == "__main__":
    a = generate_prompt("푸바오는 최근 중국에서 태어난 자이언트 판다로, 많은 주목을 받고 있습니다. 중국의 여러 동물원에서 푸바오의 성장과 건강을 지속적으로 모니터링하고 있다고 합니다.")
    print(a)
