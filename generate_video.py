import subprocess
from pathlib import Path
from typing import List
import re
import time
from datetime import datetime
import PIL
from openai import OpenAI
import shutil
import json
import asyncio
import os
import api_key

from PIL import Image, ImageFilter

client = OpenAI(
     api_key = api_key.get_gpt_key()
)

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.heif'}
ALLOWED_IMAGE_MIMETYPES = {
    'image/jpeg', 
    'image/png', 
    'image/gif', 
    'image/webp',
    'image/heic',
    'image/heif',
    'image/heic-sequence',
    'image/heif-sequence'
}
MAX_FILENAME_LENGTH = 255

def resize_and_crop(img, target_size: tuple[int, int]):
    # Resize and crop an image to the target size while preserving aspect ratio.
    # The function scales the image to fit the target size and crops the excess from the center.
    # If the image is wider than the target ratio, it scales by height and crops width.
    # If the image is taller than the target ratio, it scales by width and crops height.
    # Args:
    #     img: A PIL Image object to be resized and cropped.
    #     target_size: A tuple of (width, height) in pixels representing the desired output size.
    # Returns:
    #     A PIL Image object that has been resized and cropped to exactly match the target_size.
    # Example:
    #     >>> from PIL import Image
    #     >>> original_img = Image.open('example.jpg')
    #     >>> resized_img = resize_and_crop(original_img, (800, 600))
    """
    이미지의 비율을 유지하면서 대상 크기에 맞게 스케일링하고 중앙을 기준으로 잘라냅니다.
    """
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        # 이미지가 더 넓을 경우: 높이를 맞추고 가로를 잘라냅니다.
        scale_factor = target_size[1] / img.height
    else:
        # 이미지가 더 높거나 같을 경우: 너비를 맞추고 세로를 잘라냅니다.
        scale_factor = target_size[0] / img.width

    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    img = img.resize(new_size, Image.LANCZOS)

    # 중앙을 기준으로 잘라내기
    left = (img.width - target_size[0]) / 2
    top = (img.height - target_size[1]) / 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    img = img.crop((left, top, right, bottom))
    return img

def process_image(input_path, output_path, target_size=(720, 1280)):
    # 이미지를 열고, PNG로 변환
    with Image.open(input_path) as img:
        # 원본 이미지를 RGBA로 변환
        img = img.convert("RGBA")
        
        # 원본 이미지를 비율을 유지하면서 흐림 처리할 배경 크기에 맞게 스케일링 및 자르기
        blurred_image = resize_and_crop(img, target_size).filter(ImageFilter.GaussianBlur(radius=10))

        # 흐린 배경을 만들기
        background = Image.new("RGBA", target_size, (255, 255, 255, 255))  # 흰색 배경
        background.paste(blurred_image, (0, 0))  # 흐린 이미지로 배경을 채움

        # 원본 이미지를 비율에 맞춰 크기 조정
        img.thumbnail(target_size, Image.LANCZOS)

        # 배경을 흐리게 처리 후 크기를 맞추어 합성
        bg_width, bg_height = target_size
        img_width, img_height = img.size

        # 이미지를 가운데에 배치
        left = (bg_width - img_width) // 2
        top = (bg_height - img_height) // 2
        background.paste(img, (left, top), img)  # 투명도 처리하여 배치

        # 최종 이미지를 PNG으로 저장
        background.save(output_path, "PNG")

def process_all_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp')):  # 이미지 확장자 체크
            input_path = os.path.join(folder_path, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(folder_path, output_filename)
            process_image(input_path, output_path)
            print(f"Processed: {filename} -> {output_filename}")

TEMP_DIR_BASE = Path(os.getcwd()) / "temp_storage"
TEMP_DIR_BASE.mkdir(exist_ok=True)

async def get_audio_duration(file_path: str) -> float:
    """Get audio file duration using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        str(file_path)
    ]
    
    print(f"\n[FFprobe] Executing command: {' '.join(cmd)}")
    
    # 비동기 서브프로세스 실행
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    
    print(f"[FFprobe] stdout: {stdout_str}")
    if stderr_str:
        print(f"[FFprobe] stderr: {stderr_str}")
    
    if process.returncode != 0:
        raise Exception(f"Failed to get audio duration: {stderr_str}")
        
    data = json.loads(stdout_str)
    duration = float(data['format']['duration'])
    print(f"[FFprobe] Extracted duration: {duration} seconds")
    return duration

async def process_files(images_file: list, images_caption: list) -> str:
    """Process image and caption files to create a video"""
    print("Starting process_files function")  # 함수 시작 로그
    
    # 입력 유효성 검사
    print(f"Received {len(images_file)} images and {len(images_caption)} captions")
    if len(images_file) != len(images_caption):
        return "Number of images and captions must match"
    
    # 이미지 파일 검증
    for file in images_file:
        print(f"Validating file: {file.filename}")  # 각 파일 검증 로그

    # Make TEMP directory
    print(f"Creating temp directory at: {TEMP_DIR_BASE}")  # 디렉토리 생성 로그
    TEMP_DIR_BASE.mkdir(exist_ok=True)
    timestamp = int(time.time())
    temp_dir = TEMP_DIR_BASE / f"task_{timestamp}"
    temp_dir.mkdir(exist_ok=True)
    
    # 1. Save image files
    original_image_paths = []
    image_paths = []
    for i, file in enumerate(images_file):
        try:
            ext = Path(file.filename).suffix.lower()
            safe_filename = sanitize_filename(f"image_{i}{ext}")
            image_path = temp_dir / safe_filename
            print(f"Saving file {file.filename} to {image_path}")  # 파일 저장 로그
            
            content = await file.read()
            with open(image_path, "wb") as f:
                f.write(content)
            original_image_paths.append(image_path)

        except Exception as e:
            print(f"Error processing file {file.filename}: {str(e)}")  # 에러 로그
            raise

    for input_path in original_image_paths:
        output_filename = Path(input_path).stem + "-blur.png"
        output_path = os.path.join(temp_dir, output_filename)
        print(f"Processing image: {input_path} -> {output_path}")  # 이미지 처리 로그
        process_image(input_path, output_path)
        image_paths.append(output_path)

    # 2. Generate audio files using gTTS and get their durations
    audio_paths = []
    audio_durations = []
    for i, caption in enumerate(images_caption):
        audio_path = temp_dir / f"audio_{i}.mp3"
        print(f"audio_path: {audio_path}")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=caption
        )

        response.stream_to_file(str(audio_path))
        duration = await get_audio_duration(str(audio_path))
        print(f"duration: {duration}")
        audio_paths.append(audio_path)
        audio_durations.append(duration)
    print(f"audio_paths Done!")
    # 3. Create images.txt with matching durations
    images_txt_content = ""
    print(f"images_txt_content")
    print(f"image_paths: {image_paths}")
    print(f"audio_durations: {audio_durations}")
    for image_path, duration in zip(image_paths, audio_durations):
        print(f"images_txt_content: {images_txt_content}")
        print(f"image_path: {image_path}")
        print(f"duration: {duration}")
        image_path_name = image_path.split("/")[-1]
        images_txt_content += f"file '{image_path_name}'\nduration {duration}\n"

    images_txt_path = temp_dir / "images.txt"
    with open(images_txt_path, "w") as f:
        f.write(images_txt_content)
    print(f"images_txt_path: {images_txt_path}")

    # 4. Create subtitles.ass with matching timings
    subtitles_content = """[Script Info]
; Script generated by Python
Title: Generated ASS Subtitle
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,15,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,1,0,2,10,10,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    current_time = 0
    for caption, duration in zip(images_caption, audio_durations):
        end_time = current_time + duration
        subtitles_content += f"Dialogue: 0,{format_time(current_time)},{format_time(end_time)},Default,,0,0,0,,{caption}\n"
        current_time = end_time
        print(f"subtitles_content: {subtitles_content}")

    subtitles_path = temp_dir / "subtitles.ass"
    with open(subtitles_path, "w", encoding='utf-8') as f:
        f.write(subtitles_content)
    print(f"subtitles_path: {subtitles_path}")

    # 5. Create audio.txt
    audio_txt_content = ""
    for audio_path in audio_paths:
        audio_txt_content += f"file '{audio_path.name}'\n"
        print(f"audio_txt_content: {audio_txt_content}")
    audio_txt_path = temp_dir / "audio.txt"
    with open(audio_txt_path, "w") as f:
        f.write(audio_txt_content)
    print(f"audio_txt_path: {audio_txt_path}")

    # 6. Generate output filename using timestamp
    output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = TEMP_DIR_BASE / output_filename
    print(f"output_path: {output_path}")
    # FFmpeg command
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(images_txt_path.resolve()),
        "-f", "concat",
        "-safe", "0",
        "-i", str(audio_txt_path.resolve()),
        "-vf", f"fps=30,format=yuv420p,subtitles={str(subtitles_path.resolve())}",
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        "-c:a", "copy",
        "-progress", "pipe:1",
        str(output_path.resolve())
    ]

    # run subprocess - shell=False to prevent command injection
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1024*1024  # Limit Output Buffer
    )

    # Read output from the process
    async for line in process.stdout:
        print(line.decode().strip())


    return output_filename


# 시간 포맷팅을 위한 헬퍼 함수
def format_time(seconds: float) -> str:
    """Convert seconds to ASS time format (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    # 소수점 두 자리까지 표시
    return f"{hours}:{minutes:02d}:{secs:06.2f}"

def sanitize_filename(filename: str) -> str:
    """안전한 파일명으로 변환"""
    # 확일 확장자 보존을 위해 분리
    name, ext = os.path.splitext(filename)
    
    # 위험할 수 있는 문자들 제거 (경로 구분자, 특수문자 등)
    sanitized_name = re.sub(r'[\\/*?:"<>|]', '', name)
    
    # 공백은 언더스코어로 변경
    sanitized_name = sanitized_name.replace(' ', '_')
    
    # 확장자 다시 붙이기
    return sanitized_name + ext

def validate_image_file(file) -> bool:
    """이미지 파일 유효성 검증"""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False
    if file.content_type not in ALLOWED_IMAGE_MIMETYPES:
        return False
    return True


