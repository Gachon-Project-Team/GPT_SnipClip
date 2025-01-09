import generate_script
import generate_image
import generate_scrap
import json
 
def execute(query):
  # news=generate_scrap.news_scraper(query)
  # with open ('scrap.json', 'r', encoding='utf-8') as file:
  #   news = json.load(file)
  # script=generate_script.generate_script(news, query)
  # with open ('script.json', 'w', encoding='utf-8') as file:
  #   json.dump(script, file,  ensure_ascii=False, indent=4)
  with open ('script.json', 'r', encoding='utf-8') as file:
    script = json.load(file)
  result = generate_image.generate_image(script, query)
  with open ('result.json', 'w', encoding='utf-8') as file:
    json.dump(result, file,  ensure_ascii=False, indent=4)
  
  # with open ('script.json', 'r', encoding='utf-8') as file:
  #   script = json.load(file)
  # result=generate_image.generate_image(script, ai)
  
  

execute("가천대학교") 