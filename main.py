import generate_script
import generate_image
import generate_scrap
import json

#지금 분류만 3.5모델 씀 

def execute(query, ai):
  news=generate_scrap.news_scraper(query)
  # with open ('scrap.json', 'w', encoding='utf-8') as file:
  #   json.dump(news, file,  ensure_ascii=False, indent=4)
  script=generate_script.generate_script(news, query)
  # with open ('script.json', 'w', encoding='utf-8') as file:
  #   json.dump(script, file,  ensure_ascii=False, indent=4)
  result=generate_image.generate_image(news, script, ai)
  # with open ('use_real_image.json', 'w', encoding='utf-8') as file:
  #   json.dump(result, file,  ensure_ascii=False, indent=4)
  
  return result

execute("가천대학교", 0)