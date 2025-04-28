import generate_scrap
import generate_script
import generate_real_image
import generate_ai_image
import json

def execute(query):
    # 뉴스 스크래핑 및 스크립트 생성
    news = generate_scrap.news_scraper(query)
    script = generate_script.generate_script(news, query)
    
    # 실제 이미지 버전
    real_result = generate_real_image.generate_real_image(script, query)
    with open("result.json", "w", encoding="utf-8") as json_file:
        json.dump(real_result, json_file, indent=4, ensure_ascii=False)

    '''# AI 이미지 버전
    ai_result = generate_ai_image.generate_ai_image(script, query)
    with open("ai_result.json", "w", encoding="utf-8") as json_file:
        json.dump(ai_result, json_file, indent=4, ensure_ascii=False)'''
    
    return {
        "real_result": real_result,
        #"ai_result": ai_result,
    }

if __name__ == "__main__":
    result = execute("SKT 해킹")