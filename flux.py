import os
import torch
from diffusers import FluxPipeline

# 이미지 저장 디렉토리 생성
IMAGE_SAVE_DIR = "generated_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# FLUX.1 파이프라인 설정 함수
def set_up_flux():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe

# FLUX.1 이미지 생성 함수
def execute_flux(prompt):
    pipe = set_up_flux()

    result = []
    for prompt_data in prompt:
        category = prompt_data["category"]
        prompt_list = prompt_data["prompt"]

        img_urls = []
        for prompt in prompt_list:
            try:
                image = pipe(
                    prompt,
                    height=512,
                    width=512,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]

                filename = f"{category}_{hash(prompt)}.png"
                image.save(filename)
                inner_list=[]
                inner_list.append(filename)
                img_urls.append(inner_list)

            except Exception as e:
                print(f"* generate_image / execute_Flux * Error generating image for prompt '{prompt}': {e}")

        torch.cuda.empty_cache()
        result.append({
            "category": category,
            "image": img_urls
        })

    return result