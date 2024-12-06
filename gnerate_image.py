import json

def extracted_img(news):
    result = []
    for category, articles in news.items():
        category_data = {
            "category": category,
            "image": [(article["image"], article["url"]) for article in articles]
        }
        result.append(category_data)
    return result


def merge_script_image(script, image):
    # 1. 이미지 데이터를 category 기준으로 매핑
    image_mapping = {item["category"]: item["image"] for item in image}

    # 2. 스크립트 데이터를 순회하며 병합
    result = []
    for item in script:
        merged_item = {
            "category": item["category"],
            "title": item["title"],
            "content": item["content"],
            "image": image_mapping.get(item["category"], None) 
        }
        result.append(merged_item)

    return result


#메인 함수 
def generate_image(news, script, ai): 
  #카테고리 별 이미지랑 url만 추출
  image_url = extracted_img(news)
  image = merge_script_image(script, image_url)
  
  #ai 사용 여부에 따라 result 생성
  #ai 사용하면 스크립트에 맞춘 그림 전부 생성 
  if (ai) : 
    print("fail")  
  #ai 사용 안하면 실제 기사에 있던 이미지만 사용
  else: 
    return image
  
#test code 
#with open('scrap.json', 'r', encoding='utf-8') as file:
#            news = json.load(file)
#with open('scrip.json', 'r', encoding='utf-8') as file:
#            script = json.load(file)
#
#final = generate_image(news, script, 0)
#print(final)