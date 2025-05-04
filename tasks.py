from celery_app import celery_app
from generate_real_image import generate_real_image
from generate_ai_image import generate_ai_image
from utils.save_json import save_json_result
from generate_video import process_files


@celery_app.task
def run_real_image_matching(script, query):
    result = generate_real_image(script, query)
    save_json_result(result, query, "image")
    return result


@celery_app.task
def run_ai_image_generation(script, query):
    result = generate_ai_image(script, query)
    save_json_result(result, query, "ai_image")
    return result