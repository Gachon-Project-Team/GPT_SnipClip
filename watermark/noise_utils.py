import numpy as np
'''
1. shape : 생성할 노이즈 배열 크기
2. mean : 가우시안 분포 평균값
3. std : 가우시안 분포 표준편차
4. seed : 난수 생성 시드 값
'''
def generate_noise(shape, mean=0, std=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(mean, std, shape).astype(np.float32)
    return noise
