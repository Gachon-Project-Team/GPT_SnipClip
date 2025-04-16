import os
import json
from datetime import datetime

def save_json_result(result, query, step):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query.replace(" ", "_").replace("/", "_")  # 경로 방지
    dir_path = f"data/{step}"
    os.makedirs(dir_path, exist_ok=True)

    filename = f"{timestamp}_{safe_query}.json"
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return full_path