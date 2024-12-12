import generate_script
import generate_image
import generate_scrap
import json

def execute(query, ai):
  news=generate_scrap.news_scraper(query)
  with open ('scrap.json', 'w', encoding='utf-8') as file:
    json.dump(news, file,  ensure_ascii=False, indent=4)
  script=generate_script.generate_script(news, query)
  with open ('script.json', 'r', encoding='utf-8') as file:
    script=json.load(file)
  result=generate_image.generate_image(script, ai)
  with open ('use_ai_image.json', 'w', encoding='utf-8') as file:
    json.dump(result, file,  ensure_ascii=False, indent=4)
  
execute("가천대학교", 1)