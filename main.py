import generate_script
import generate_image
import generate_scrap
from datetime import datetime
import json
 
def execute(query):
  news=generate_scrap.news_scraper(query)
  with open("scrap.json", "w", encoding="utf-8") as json_file:
     json.dump(news, json_file, indent=4, ensure_ascii=False)
  with open("scrap.json", "r", encoding="utf-8") as json_file:
      news = json.load(json_file) 
  script=generate_script.generate_script(news, query)
  with open("script.json", "w", encoding="utf-8") as json_file:
     json.dump(script, json_file, indent=4, ensure_ascii=False)

execute("가천대") 
