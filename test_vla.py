import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
import time
import numpy as np
# 1. 설정 (Configuration)
# AGX Thor는 bfloat16을 네이티브로 지원하며 가장 빠릅니다.
MODEL_ID = "openvla/openvla-7b"
DEVICE = "cuda"
DTYPE = torch.bfloat16 

print(f"--- [OpenVLA on Thor] 시작합니다 ---")
print(f"사용 장치: {torch.cuda.get_device_name(0)}")
print(f"정밀도: {DTYPE}")

# 2. 모델 및 프로세서 로드
print("1. 모델 다운로드 및 로딩 중... (처음 실행 시 시간이 걸립니다)")
start_time = time.time()

# 주의: trust_remote_code=True 필수 (OpenVLA는 커스텀 모델 아키텍처 사용)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # Flash Attention 2 강제 적용 (우리가 힘들게 빌드한 이유!)
    #use_flash_attention_2=True 
    attn_implementation="sdpa"
).to(DEVICE)

print(f"-> 모델 로드 완료! (소요 시간: {time.time() - start_time:.2f}초)")

# 3. 테스트용 이미지 준비 (인터넷에서 가져옴)
print("2. 테스트 이미지 준비 중...")

url = "https://upload.wikimedia.org/wikipedia/commons/6/6d/Bananas_white_background_DS.jpg"

try:
    resp = requests.get(url, stream=True, timeout=5)
    resp.raise_for_status() # 404 에러 시 예외 발생
    image = Image.open(resp.raw).convert("RGB")
    print("-> 인터넷 이미지 다운로드 성공!")
    prompt_text = "Pick up the banana" # 바나나 이미지에 맞는 프롬프트

except Exception as e:
    print(f"-> 이미지 다운로드 실패 ({e}). 임시 이미지를 생성합니다.")
    # 인터넷 안 될 경우: 랜덤 노이즈 이미지 생성 (모델 작동 확인용)
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    prompt_text = "Pick up the object" 

# 4. 프롬프트 작성 (이미지에 맞춰 자동 변경)
instruction = prompt_text
prompt = f"In: {instruction} [IMAGE] Out:"

# 5. 입력 전처리
inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE, dtype=DTYPE)

# 6. 추론 (Action Generation)
print("3. 로봇 액션 생성 중 (Inference)...")
start_time = time.time()

with torch.inference_mode():
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

end_time = time.time()
print(f"-> 추론 완료! (소요 시간: {end_time - start_time:.4f}초)")

# 7. 결과 출력
# OpenVLA는 7자유도(x, y, z, r, p, y, gripper) 액션을 반환합니다.
print("\n=== 예측된 로봇 액션 (Normalized Action) ===")
print(action)
print("==========================================")