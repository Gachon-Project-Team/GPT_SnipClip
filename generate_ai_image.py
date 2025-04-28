import json
import torch
import gc
import time
import logging
from flux import generate_prompt, execute_flux
from generate_script import generate_script

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_section_groups(sections, num_groups=5):
    if not sections or num_groups <= 0:
        return []
    if len(sections) <= num_groups:
        return [[section] for section in sections]

    sections_per_group = len(sections) // num_groups
    remainder = len(sections) % num_groups

    groups, start_idx = [], 0
    for i in range(num_groups):
        count = sections_per_group + (1 if i < remainder else 0)
        end_idx = start_idx + count
        groups.append(sections[start_idx:end_idx])
        start_idx = end_idx

    return groups

def generate_ai_image(news, query):
    start_time = time.time()  # 전체 프로세스 시작 시간
    logging.info(f"'{query}' AI 이미지 생성 시작")
    
    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if isinstance(news, list):
        logging.info("뉴스가 리스트 형태입니다. 'default' 카테고리로 변환합니다.")
        news = {"default": news}
    
    # 스크립트 생성 시간 측정
    script_start_time = time.time()
    try:
        script_data = generate_script(news, query, use_real_image=False)
        script_elapsed_time = time.time() - script_start_time
        logging.info(f"스크립트 생성 완료: {script_elapsed_time:.2f}초 소요")
    except Exception as e:
        script_elapsed_time = time.time() - script_start_time
        logging.error(f"generate_script 실패: {e} ({script_elapsed_time:.2f}초 소요)")
        return []
    
    if len(script_data) == 1 and isinstance(script_data[0], list):
        script_data = script_data[0]
    
    # 스크립트 통계
    total_sections = sum(len(create_section_groups(item.get("section", []))) for item in script_data if isinstance(item, dict))
    logging.info(f"총 {total_sections}개 섹션에 대한 AI 이미지 생성 예정")
    
    results = []
    actual_total_sections = 0
    total_ai_images = 0
    
    # 프롬프트 생성 및 이미지 생성 총 시간 측정을 위한 변수
    total_prompt_generation_time = 0
    total_image_generation_time = 0
    
    # 각 스크립트 항목(카테고리)에 대해 처리
    for item_idx, item in enumerate(script_data):
        if not isinstance(item, dict):
            continue
            
        category = item.get("category", "unknown")
        title = item.get("title", category)
        sections = item.get("section") or item.get("sections") or []
        
        if not sections:
            logging.warning(f"스크립트 항목 {item_idx+1}에 섹션 없음: {title}")
            continue
            
        category_start_time = time.time()
        logging.info(f"처리 중: 카테고리 '{category}', {len(sections)}개 섹션")
        
        # 섹션 그룹화 (더 효율적인 이미지 매칭을 위해)
        section_groups = create_section_groups(sections, 5)
        matched_images = []
        category_ai_images = 0
        
        # 각 섹션 그룹에 대한 이미지 생성
        for group_idx, section_group in enumerate(section_groups):
            if not section_group:
                continue
                
            actual_total_sections += 1
            combined_text = " ".join(section_group)
            
            # 프롬프트 생성 시간 측정
            prompt_start_time = time.time()
            prompt = generate_prompt(combined_text)
            prompt_generation_time = time.time() - prompt_start_time
            total_prompt_generation_time += prompt_generation_time
            
            logging.info(f"섹션 그룹 {group_idx+1}/{len(section_groups)} 처리 중")
            logging.info(f"  - 프롬프트 생성 완료: {prompt_generation_time:.2f}초 소요")
            logging.info(f"  - 생성된 프롬프트: {prompt[:100]}...")
            
            # AI 이미지 생성 시간 측정
            image_start_time = time.time()
            ai_url = execute_flux(prompt)
            image_generation_time = time.time() - image_start_time
            total_image_generation_time += image_generation_time
            
            if ai_url:
                matched_images.append(ai_url)  # AI URL만 저장
                total_ai_images += 1
                category_ai_images += 1
                logging.info(f"  - 이미지 생성 성공: {image_generation_time:.2f}초 소요")
            else:
                logging.warning(f"  - 이미지 생성 실패: {image_generation_time:.2f}초 소요")
            
            # 메모리 정리
            memory_cleanup_start = time.time()
            torch.cuda.empty_cache()
            gc.collect()
            memory_cleanup_time = time.time() - memory_cleanup_start
            logging.info(f"  - 메모리 정리: {memory_cleanup_time:.2f}초 소요")
            
            # 요청 간 간격 조정 (API 제한 방지)
            time.sleep(1)
        
        category_elapsed_time = time.time() - category_start_time
        logging.info(f"카테고리 '{category}' 처리 완료: {len(matched_images)}개 이미지 생성 ({category_elapsed_time:.2f}초 소요)")
        
        result_item = {
            "category": category,
            "title": title,
            "section": sections,
            "image": matched_images
        }
        results.append(result_item)
    
    # 결과 저장 시간 측정
    save_start_time = time.time()
    save_elapsed_time = time.time() - save_start_time
    
    # 전체 시간 및 세부 시간 통계
    total_elapsed_time = time.time() - start_time
    
    # 통계 출력
    logging.info(f"AI 이미지 생성 완료: {total_elapsed_time:.2f}초 소요")
    logging.info(f"- 스크립트 생성: {script_elapsed_time:.2f}초 ({script_elapsed_time/total_elapsed_time*100:.1f}%)")
    logging.info(f"- 프롬프트 생성: {total_prompt_generation_time:.2f}초 ({total_prompt_generation_time/total_elapsed_time*100:.1f}%)")
    logging.info(f"- 이미지 생성: {total_image_generation_time:.2f}초 ({total_image_generation_time/total_elapsed_time*100:.1f}%)")
    logging.info(f"- 결과 저장: {save_elapsed_time:.2f}초 ({save_elapsed_time/total_elapsed_time*100:.1f}%)")
    
    logging.info(f"이미지 생성 결과:")
    logging.info(f"- 총 섹션: {actual_total_sections}개")
    logging.info(f"- AI 이미지: {total_ai_images}개 ({100*total_ai_images/max(1,actual_total_sections):.1f}%)")
    logging.info(f"- 평균 이미지 생성 시간: {total_image_generation_time/max(1,total_ai_images):.2f}초/이미지")
    
    return results