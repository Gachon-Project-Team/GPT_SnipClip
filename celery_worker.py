from celery import Celery

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",   # Redis 브로커 주소
    backend="redis://localhost:6379/0"   # 결과도 Redis에 저장
)
