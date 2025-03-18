from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import generate_scrap
import generate_script
import generate_image
import flux
import json
from typing import Dict, List
import logging

app = FastAPI()

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
    data: Dict[str, List[NewsItem]]  # 동적으로 키 이름을 처리
    query: str

class ImageRequest(BaseModel):
    script: list
    query: str

class FluxRequest(BaseModel):
    prompt: str

@app.post("/scrap")
async def scrap_news(request: QueryRequest):
    try:
        result = generate_scrap.news_scraper(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping news: {e}")

@app.post("/script")
async def generate_script_api(request: ScriptRequest):
    logger = logging.getLogger("app_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        # 요청 데이터 로깅

        logger.info(f"Received /script request: {request}")

        # 각 키와 값 처리
        results = {}
        for key, news_list in request.data.items():
            logger.info(f"Processing key: {key} with {len(news_list)} items")
            results[key] = generate_script.generate_script(news_list, request.query)

        # 결과 반환
        return {"message": "Scripts generated successfully", "results": results}
    except Exception as e:
        logger.error(f"Error in /script: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating script: {e}")

@app.post("/image")
async def generate_image_api(request: ImageRequest):
    try:
        result = generate_image.generate_image(request.script, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {e}")

@app.post("/flux")
async def generate_flux_image(request: FluxRequest):
    try:
        result = flux.execute_flux(request.prompt)
        return {"image_url": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating flux image: {e}")

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
        raise HTTPException(status_code=500, detail=f"Error executing pipeline: {e}")