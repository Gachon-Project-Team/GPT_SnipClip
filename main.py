import generate_script
import generate_image
import generate_scrap
from datetime import datetime
import json
 
def execute(query):
  news=generate_scrap.news_scraper(query)
  script=generate_script.generate_script(news, query)
  result = generate_image.generate_image(script, query)

  return result

result = execute("query") 
with open("result.json", "w") as json_file:
    json.dump(result, json_file, indent=4)