from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_scrap_news():
    response = client.post("/scrap", json={"query": "가천대학교"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_generate_script():
    news = [{"id": 1, "title": "Test News", "content": "Test Content", "image": "url", "url": "url"}]
    response = client.post("/script", json={"news": news, "query": "가천대학교"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_generate_image():
    script = [{"category": "Test", "sections": ["Section 1", "Section 2"]}]
    response = client.post("/image", json={"script": script, "query": "가천대학교"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_generate_flux_image():
    response = client.post("/flux", json={"prompt": "A panda in a forest"})
    assert response.status_code == 200
    assert "image_url" in response.json()