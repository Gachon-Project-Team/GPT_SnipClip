import generate_script
import generate_image
import generate_scrap
import json
 
def execute(query):
   news=generate_scrap.news_scraper(query)
   script=generate_script.generate_script(news, query)
   result = generate_image.generate_image(script, query)
   with open("result.json", "w", encoding="utf-8") as json_file:
      json.dump(result, json_file, indent=4, ensure_ascii=False)
      
execute("검색어") 
