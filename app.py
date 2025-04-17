from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List
from utils.save_json import save_json_result
import generate_scrap
import generate_script
import generate_image
import generate_video
import flux
import json
import os

app = FastAPI()

# 정적 파일 제공 설정
app.mount("/videos", StaticFiles(directory="temp_storage"), name="videos")
app.mount("/generated_images",
          StaticFiles(directory="generated_images"), name="generated_images")
# 브라우저에서 /image 의 결과인 실제 이미지 url 을 파일 형태로 변환 후 /video 로 요청할 시 CORS 에러 발생. /image 의 결과는 생성 혹은 다운받은 실제 이미지의 서버 경로를 반환하도록 한다
# app.mount("/image_files", StaticFiles(directory="image"), name="image_files")
app.mount("/image", StaticFiles(directory="image"), name="image")
# 요청 데이터 모델 정의


class QueryRequest(BaseModel):
    query: str


class NewsItem(BaseModel):
    id: int
    title: str
    content: str
    image: str
    url: str


class ScriptRequest(BaseModel):
    news: Dict[str, List[NewsItem]]  # 동적으로 키 이름을 처리
    query: str


class ImageRequest(BaseModel):
    script: list
    query: str


class FluxRequest(BaseModel):
    prompt: str
    client_ip: str = "127.0.0.1"
    width: int = 1280
    height: int = 720
    guidance_scale: float = 0.5
    num_inference_steps: int = 100


@app.post("/scrap")
async def scrap_news(request: QueryRequest):
    try:
        result = generate_scrap.news_scraper(request.query)

        save_json_result(result, request.query, "scrap")

        return {"news": result, "query": request.query}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error scraping news: {e}")


@app.post("/script")
async def generate_script_api(request: ScriptRequest):
    try:
        # 요청 데이터 디버깅
        print(f"Type of request.news: {type(request.news)}")
        print(f"Content of request.news: {request.news}")

        # generate_script 함수에 전달할 데이터 준비
        formatted_news = {}
        for key, news_list in request.news.items():
            formatted_news[key] = [news_item.dict() for news_item in news_list]

        print(f"Formatted news: {formatted_news}")

        # generate_script 호출
        results = generate_script.generate_script(
            formatted_news, request.query)

        save_json_result(results, request.query, "script")

        return results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating script: {e}")


@app.post("/image")
async def generate_image_api(request: ImageRequest):
    try:
        result = generate_image.generate_image(request.script, request.query)

        save_json_result(result, request.query, "image")

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating image: {str(e)}")


@app.post("/flux")
async def generate_flux_image(request: FluxRequest):
    """
    Generates an image using the flux.execute_flux function.
    """
    try:
        # flux.execute_flux 호출
        image_url = flux.execute_flux(
            prompt=request.prompt,
            client_ip=request.client_ip,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps
        )

        if not image_url:
            raise HTTPException(
                status_code=500, detail="Failed to generate image")

        return image_url
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating flux image: {e}")


@app.post("/execute")
async def execute_pipeline(request: QueryRequest):
    """
    Executes the full pipeline: scraping news, generating a script, and creating images.
    """
    try:
        # Step 1: Scrape news
        news = generate_scrap.news_scraper(request.query)
        with open("scrap.json", "w", encoding="utf-8") as json_file:
            json.dump(news, json_file, indent=4, ensure_ascii=False)

        # Step 2: Generate script
        script = generate_script.generate_script(news, request.query)
        with open("script.json", "w", encoding="utf-8") as json_file:
            json.dump(script, json_file, indent=4, ensure_ascii=False)

        # Step 3: Generate images
        result = generate_image.generate_image(script, request.query)
        with open("result.json", "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)

        # Return the final result as JSON
        return {"message": "Pipeline executed successfully", "result": result}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error executing pipeline: {e}")


@app.post("/video")
async def generate_video_api(
    images: List[UploadFile],  # 이미지 파일 리스트
    captions: List[str] = Form(...)  # 캡션 리스트
):
    """
    API to generate a video from images and captions.
    """
    try:
        # 입력 유효성 검사
        if len(images) != len(captions):
            raise HTTPException(
                status_code=400,
                detail="Number of images and captions must match"
            )

        # process_files 호출
        output_filename = await generate_video.process_files(images, captions)

        # 비디오 파일 경로 생성
        video_url = f"/videos/{output_filename}"

        # 결과 반환
        return JSONResponse(
            content={
                "message": "Video generated successfully",
                "video_url": video_url
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating video: {e}")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
