import os
import datetime
import asyncio
import paramiko
import api_key
import json
import requests
import subprocess

# FLUX.1 이미지 생성 함수 - AI 서버에 접속
def execute_flux(prompt, client_ip='127.0.0.1', width=1280, height=720, guidance_scale=0.5, num_inference_steps=100):
    try:
        # 환경 변수 및 설정 가져오기
        CONDA_ENV_NAME = api_key.get_CONDA_ENV_NAME()
        GENERATE_SCRIPT = api_key.get_GENERATE_SCRIPT()
        SSH_HOST = api_key.get_SSH_HOST()
        SSH_PORT = api_key.get_SSH_PORT()
        SSH_USERNAME = api_key.get_SSH_USERNAME()
        SSH_PASSWORD = api_key.get_SSH_PASSWORD()
        OUTPUT_DIR = api_key.get_OUTPUT_DIR().replace("\"", "").replace("'", "")
        LOCAL_SAVE_DIR = api_key.get_LOCAL_SAVE_DIR()

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

        print(f"Processing complete. Image available at /generated_images/{output_filename}")
        image_url = '/generated_images/' + output_filename

    except Exception as e:
        print(f"* generate_image / execute_Flux * Error generating image for prompt '{prompt}': {e}")
        image_url = None

    finally:
        # SSH 연결 종료
        if 'ssh' in locals():
            ssh.close()

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
