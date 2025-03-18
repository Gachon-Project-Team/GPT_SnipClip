from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import generate_scrap
import generate_script
import generate_image
import flux

app = FastAPI()

# 요청 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str

class ScriptRequest(BaseModel):
    news: list
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
    try:
        result = generate_script.generate_script(request.news, request.query)
        return result
    except Exception as e:
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