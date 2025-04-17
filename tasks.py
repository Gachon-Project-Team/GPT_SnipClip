from celery_worker import celery_app
from generate_image import generate_image
from utils.save_json import save_json_result
from generate_video import process_files

@celery_app.task
def run_image_generation(script, query):
    result = generate_image(script, query)
    save_json_result(result, query, "image")
    return result

