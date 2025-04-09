import os
import torch
import api_key
import json
import requests
import hashlib
import datetime
import asyncio
import paramiko
import subprocess
import re

# FLUX.1 이미지 생성 함수 - AI 서버에 접속
def execute_flux(prompt, client_ip='127.0.0.1', width=1280, height=720, guidance_scale=0.5, num_inference_steps=100):
    HOST = None
    image_url = None
    try:
        # 환경 변수 및 설정 가져오기
        HOST = api_key.get_HOST()
        CONDA_ENV_NAME = api_key.get_CONDA_ENV_NAME()
        GENERATE_SCRIPT = api_key.get_GENERATE_SCRIPT()
        SSH_HOST = api_key.get_SSH_HOST()
        SSH_PORT = api_key.get_SSH_PORT()
        SSH_USERNAME = api_key.get_SSH_USERNAME()
        SSH_PASSWORD = api_key.get_SSH_PASSWORD()
        OUTPUT_DIR = api_key.get_OUTPUT_DIR().replace("\"", "").replace("'", "")
        LOCAL_SAVE_DIR = api_key.get_LOCAL_SAVE_DIR().replace("\"", "").replace("'", "")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_ip = client_ip.replace(".", "-")  # IP 주소에서 dot을 하이픈으로 변경
        output_filename = f"{timestamp}_{sanitized_ip}.png"

        # 실행할 명령어
        command = f'source /home/{SSH_USERNAME}/anaconda3/bin/activate {CONDA_ENV_NAME} && '
        command += f'python3 {GENERATE_SCRIPT} --prompt "{prompt}" --guidance_scale {guidance_scale} '
        command += f'--num_inference_steps {num_inference_steps} --width {width} --height {height} '
        command += f'--output {os.path.join(OUTPUT_DIR, output_filename)}'

        print(f"Executing remote command via SSH: {command}")

        # SSH 연결 설정
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, password=SSH_PASSWORD)
        print("SSH connection established successfully.")

        # 원격 명령어 실행
        transport = ssh.get_transport()
        channel = transport.open_session()
        channel.exec_command(command)

        # 실시간 출력 읽기
        print("Command execution started. Streaming output:")
        while True:
            if channel.recv_ready():
                stdout_line = channel.recv(1024).decode('utf-8')
                print(stdout_line, end="")  # 실시간 출력
            if channel.recv_stderr_ready():
                stderr_line = channel.recv_stderr(1024).decode('utf-8')
                print(stderr_line, end="")  # 실시간 에러 출력
            if channel.exit_status_ready():
                break

        # 명령어 종료 상태 확인
        exit_status = channel.recv_exit_status()

        # SFTP를 사용하여 생성된 파일 다운로드
        print("Attempting to retrieve the latest generated image via SFTP...")
        sftp = ssh.open_sftp()
        remote_file_path = os.path.join(OUTPUT_DIR, output_filename)
        local_path = os.path.join(LOCAL_SAVE_DIR, output_filename)
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
        try:
            sftp.get(remote_file_path, local_path)
            print(f"Successfully downloaded: {output_filename}")
        except FileNotFoundError:
            print(f"Generated image not found: {remote_file_path}")
            raise FileNotFoundError(f"Generated image not found: {remote_file_path}")

        print(f"Processing complete. Image available at /{output_filename}")
        image_url = output_filename

    except Exception as e:
        print(f"* generate_image / execute_Flux * Error generating image for prompt '{prompt}': {e}")
        image_url = None

    finally:
        # SSH 연결 종료
        if 'ssh' in locals():
            ssh.close()

    if HOST and image_url:
        return HOST + image_url
    else:
        return None


#4모델 쓸 때 반환 결과 추가 처리에 사용 
def clean_gpt_response(raw_content):
    raw_content = raw_content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]  # ```json 제거
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]  # ``` 제거
    return raw_content

# 인물/캐릭터 패턴 인식 함수
def detect_character_or_person(text):
    # 자주 등장하는 캐릭터나 유명인물을 찾는 패턴
    patterns = [
        r'[가-힣\s]+(캐릭터|인물|마스코트)',  # 캐릭터 관련 키워드
        r'(푸바오|루루|에버랜드|판다|곰)',  # 유명 동물 캐릭터 예시
        r'(뽀로로|타요|뿌까|핑크퐁|아기상어|짱구|도라에몽|미키마우스|헬로키티|포켓몬)',  # 인기 캐릭터 예시 
        r'[가-힣]{1,3}\s(캐릭터|인형)',  # 이름 + 캐릭터/인형
        r'[A-Za-z\s]+(character|mascot)'  # 영문 캐릭터 관련 단어
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

#prompt 생성 gpt 쿼리 
def setup_prompt_gpt_request(section):
    # 텍스트에 캐릭터나 인물이 있는지 확인
    has_character = detect_character_or_person(section)
    
    key = api_key.get_gpt_key()
    url = "https://api.openai.com/v1/chat/completions"
    
    header = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    # 캐릭터나 인물이 있는 경우와 없는 경우에 다른 시스템 메시지 사용
    if has_character:
        system_content = (
            "You are a prompt engineer for image generation AI specializing in creating character-focused prompts."
            "Your task is to create a concise English image generation prompt from Korean text."
            "IMPORTANT: Keep the prompt under 77 tokens/characters."
            "When generating prompts for characters or specific people:"
            "1. Focus on the character's distinct features, personality, and unique style"
            "2. Use a stylized, slightly artistic approach rather than photorealistic"
            "3. Emphasize the character's iconic traits, colors, and elements"
            "4. For animal characters like Panda Fu Bao, capture their cute, cartoon-like essence"
            "5. Generated images with every prompt should follow the same consistent style"
            "Only return the final English prompt with no explanations or additional text."
        )
    else:
        system_content = (               
            "You are a prompt engineer for image generation AI specializing in creating short, effective prompts."
            "Your task is to create a concise English image generation prompt from Korean text."
            "IMPORTANT: Keep the prompt under 77 tokens/characters."
            "Use a consistent style: photorealistic, high detail, cinematic lighting."
            "Include these style elements with every prompt: photorealistic, detailed."
            "Only return the final English prompt with no explanations or additional text."
        )

    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": (
                    "do not return any description except prompt(result)"
                    f"Here is a Korean script section: \"{section}\"\n"
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
        categories = clean_content 
    else:
        print(f"* flux /execute_gpt * response Error: {response.status_code}")
        categories = None
        
    return categories

#대본 기반 prompt생성 실행  
def generate_prompt(section):
    url, header, request = setup_prompt_gpt_request(section)
    gpt_result = execute_gpt(url, header, request)
    
    # 결과 로그 출력
    has_character = detect_character_or_person(section)
    if has_character:
        print(f"[INFO] 캐릭터/인물 감지: {section[:30]}")
    else:
        print(f"[INFO] 일반 콘텐츠: {section[:30]}")
    
    return gpt_result

