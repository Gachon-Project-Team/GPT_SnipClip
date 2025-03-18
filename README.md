# SnipClip

SnipClip is a Python-based project that automates the process of generating multimedia content, including news scraping, script generation, and image creation. It uses FastAPI to expose its functionality as HTTP APIs, enabling seamless integration with other systems.

---

## Features

1. **News Scraping**: Scrapes news articles using Naver's API and processes them with the `newspaper3k` library.
2. **Script Generation**: Summarizes and organizes news articles into structured scripts using OpenAI's GPT API.
3. **Image Generation**: Matches or generates images for the scripts using AI models like Flux and CLIP.
4. **Full Pipeline Execution**: Combines all steps into a single pipeline for end-to-end multimedia content creation.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- API keys for OpenAI and Naver (stored in a `.env` file)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Gachon-Project-Team/GPT_SnipClip.git
   cd GPT_SnipClip
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the `.env` file with your API keys:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   NAVER_CLIENT_ID=your_naver_client_id
   NAVER_CLIENT_SECRET=your_naver_client_secret
   ```

---

## Usage

### Running the FastAPI Server
Start the server using `uvicorn`:
```bash
uvicorn app:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

### API Endpoints
1. **Scrape News**: `/scrap`  
   - **Method**: POST  
   - **Body**: `{ "query": "keyword" }`  
   - **Description**: Scrapes news articles related to the given keyword.

2. **Generate Script**: `/script`  
   - **Method**: POST  
   - **Body**: `{ "news": [...], "query": "keyword" }`  
   - **Description**: Generates a structured script from the provided news data.

3. **Generate Image**: `/image`  
   - **Method**: POST  
   - **Body**: `{ "script": [...], "query": "keyword" }`  
   - **Description**: Matches or generates images for the given script.

4. **Generate Flux Image**: `/flux`  
   - **Method**: POST  
   - **Body**: `{ "prompt": "description" }`  
   - **Description**: Generates an AI image based on the provided prompt.

5. **Execute Full Pipeline**: `/execute`  
   - **Method**: POST  
   - **Body**: `{ "query": "keyword" }`  
   - **Description**: Executes the full pipeline (scraping, script generation, and image creation).

---

## Testing

Run the test suite using `pytest`:
```bash
pytest test_app.py
```

---

## Project Structure

- `app.py`: FastAPI server implementation.
- `generate_scrap.py`: Handles news scraping and classification.
- `generate_script.py`: Generates scripts using OpenAI's GPT API.
- `generate_image.py`: Matches or generates images using Flux and CLIP.
- `flux.py`: Handles AI-based image generation using the Flux model.
- `main.py`: Executes the full pipeline locally.
- `.env`: Stores API keys and configuration (excluded from version control).

---

## License

This project is licensed under the MIT License.

---

### 한국어

# SnipClip

SnipClip은 뉴스 스크래핑, 스크립트 생성, 이미지 생성 등 멀티미디어 콘텐츠 제작을 자동화하는 Python 기반 프로젝트입니다. FastAPI를 사용하여 HTTP API로 기능을 제공하며, 다른 시스템과의 통합을 용이하게 합니다.

---

## 주요 기능

1. **뉴스 스크래핑**: Naver API와 `newspaper3k` 라이브러리를 사용하여 뉴스 기사를 스크래핑합니다.
2. **스크립트 생성**: OpenAI GPT API를 사용하여 뉴스 기사를 요약하고 구조화된 스크립트를 생성합니다.
3. **이미지 생성**: Flux 및 CLIP과 같은 AI 모델을 사용하여 스크립트에 적합한 이미지를 매칭하거나 생성합니다.
4. **전체 파이프라인 실행**: 모든 단계를 결합하여 멀티미디어 콘텐츠를 엔드 투 엔드로 생성합니다.

---

## 설치 방법

### 사전 준비
- Python 3.8 이상
- OpenAI 및 Naver API 키 (`.env` 파일에 저장)

### 설치 단계
1. 레포지토리 클론:
   ```bash
   git clone https://github.com/Gachon-Project-Team/GPT_SnipClip.git
   cd GPT_SnipClip
   ```

2. 가상환경 생성:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```

3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

4. `.env` 파일 설정:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   NAVER_CLIENT_ID=your_naver_client_id
   NAVER_CLIENT_SECRET=your_naver_client_secret
   ```

---

## 사용법

### FastAPI 서버 실행
`uvicorn`을 사용하여 서버를 실행합니다:
```bash
uvicorn app:app --reload
```

서버는 `http://127.0.0.1:8000`에서 실행됩니다.

### API 엔드포인트
1. **뉴스 스크래핑**: `/scrap`  
   - **메서드**: POST  
   - **요청 본문**: `{ "query": "키워드" }`  
   - **설명**: 주어진 키워드와 관련된 뉴스 기사를 스크래핑합니다.

2. **스크립트 생성**: `/script`  
   - **메서드**: POST  
   - **요청 본문**: `{ "news": [...], "query": "키워드" }`  
   - **설명**: 제공된 뉴스 데이터를 기반으로 구조화된 스크립트를 생성합니다.

3. **이미지 생성**: `/image`  
   - **메서드**: POST  
   - **요청 본문**: `{ "script": [...], "query": "키워드" }`  
   - **설명**: 주어진 스크립트에 적합한 이미지를 매칭하거나 생성합니다.

4. **Flux 이미지 생성**: `/flux`  
   - **메서드**: POST  
   - **요청 본문**: `{ "prompt": "설명" }`  
   - **설명**: 제공된 프롬프트를 기반으로 AI 이미지를 생성합니다.

5. **전체 파이프라인 실행**: `/execute`  
   - **메서드**: POST  
   - **요청 본문**: `{ "query": "키워드" }`  
   - **설명**: 전체 파이프라인(스크래핑, 스크립트 생성, 이미지 생성)을 실행합니다.

---

## 테스트

`pytest`를 사용하여 테스트를 실행합니다:
```bash
pytest test_app.py
```

---

## 프로젝트 구조

- `app.py`: FastAPI 서버 구현.
- `generate_scrap.py`: 뉴스 스크래핑 및 분류 처리.
- `generate_script.py`: OpenAI GPT API를 사용하여 스크립트 생성.
- `generate_image.py`: Flux 및 CLIP을 사용하여 이미지 매칭 또는 생성.
- `flux.py`: Flux 모델을 사용한 AI 기반 이미지 생성 처리.
- `main.py`: 로컬에서 전체 파이프라인 실행.
- `.env`: API 키 및 설정 저장 (버전 관리에서 제외).

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
