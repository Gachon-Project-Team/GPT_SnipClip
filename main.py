import generate_script
import generate_image
import generate_scrap
import json

def execute(query, ai):
  news=generate_scrap.news_scraper(query)
  # with open ('scrap.json', 'w', encoding='utf-8') as file:
  #   json.dump(news, file,  ensure_ascii=False, indent=4)
  script=generate_script.generate_script(news, query)
  # with open ('script.json', 'w', encoding='utf-8') as file:
  #   json.dump(script, file,  ensure_ascii=False, indent=4)
  generate_image.generate_image(news, script, ai)
  # with open ('use_real_image.json', 'w', encoding='utf-8') as file:
  #   json.dump(result, file,  ensure_ascii=False, indent=4)
  
execute("mma", 0)