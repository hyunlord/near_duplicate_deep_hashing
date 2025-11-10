# 🚀 Colab 빠른 시작 가이드

이미지-텍스트 멀티모달 해싱 모델을 Colab에서 실행하는 단계별 가이드입니다.

## 📋 실행 순서

### Step 1: Colab 노트북 생성 및 GPU 설정

1. Google Colab 접속: https://colab.research.google.com/
2. 새 노트북 생성
3. **런타임 > 런타임 유형 변경 > GPU 선택**

### Step 2: 저장소 클론

```python
# Colab 셀에서 실행
!git clone https://github.com/hyunlord/near_duplicate_deep_hashing.git
%cd near_duplicate_deep_hashing
```

### Step 3: 패키지 설치

```python
# 필수 패키지 설치
!pip install -q pytorch-lightning
!pip install -q transformers
!pip install -q datasets
!pip install -q albumentations
!pip install -q matplotlib

# GPU 확인
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 4: 학습 실행

```python
# 멀티모달 학습 시작
!python main_multimodal_colab.py
```

**예상 실행 시간:** 약 30-60분 (30 에포크 기준)

**학습 중 확인사항:**
- `train/total_loss`: 손실이 감소하는지 확인
- `val/64_final_score`: 검증 점수가 증가하는지 확인
- GPU 메모리: 약 8-12GB 사용 (T4 GPU 기준)

### Step 5: 학습 모니터링

```python
# 학습 로그 확인
!tail -n 20 logs_multimodal/multimodal_hashing/version_0/metrics.csv
```

### Step 6: 학습된 모델로 검색 테스트

```python
# 모델 로드 및 검색 테스트
from app.module_multimodal import MultiModalHashingModel
from app.xor_search import MultiModalRetrieval
from transformers import AutoTokenizer
import torch

# 1. 모델 로드
checkpoint_path = './checkpoints_multimodal/last.ckpt'
model = MultiModalHashingModel.load_from_checkpoint(checkpoint_path)
model.eval()

# 2. 토크나이저
tokenizer = AutoTokenizer.from_pretrained('google/siglip-base-patch16-384')

# 3. 검색 시스템 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
retriever = MultiModalRetrieval(
    model=model,
    tokenizer=tokenizer,
    bit_length=64,
    device=device
)

print("✓ 모델 로드 완료!")
```

### Step 7: 실제 검색 예제

#### 7-1. 텍스트로 이미지 검색

```python
from PIL import Image
import requests
from io import BytesIO

# 예시 이미지 DB 준비 (인터넷에서 다운로드)
image_urls = [
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",  # cat
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1",  # dog
    "https://images.unsplash.com/photo-1426604966848-d7adac402bff",  # nature
]

images = []
for url in image_urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    images.append(img)

print(f"이미지 DB 준비 완료: {len(images)}개")

# 텍스트로 검색
query_text = "a cute cat"
indices, distances = retriever.search_image_by_text(
    query_text=query_text,
    image_database=images,
    top_k=3
)

print(f"\nQuery: '{query_text}'")
print(f"Top-3 결과:")
for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
    print(f"  {rank}. 이미지 #{idx.item()}, Hamming 거리: {dist.item()}")
```

#### 7-2. 이미지로 텍스트 검색

```python
# 텍스트 DB 준비
text_database = [
    "a cat sitting on a chair",
    "a dog playing in the park",
    "beautiful mountain landscape",
    "a bird flying in the sky",
    "a fish swimming in water"
]

# 이미지로 검색
query_image = images[0]  # 첫 번째 이미지 사용
indices, distances = retriever.search_text_by_image(
    query_image=query_image,
    text_database=text_database,
    top_k=3
)

print(f"\nQuery: 이미지 #0")
print(f"Top-3 결과:")
for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
    print(f"  {rank}. '{text_database[idx.item()]}' (Hamming 거리: {dist.item()})")
```

#### 7-3. 검색 결과 시각화

```python
import matplotlib.pyplot as plt

# 텍스트 검색 결과 시각화
query_text = "a cat"
indices, distances = retriever.search_image_by_text(query_text, images, top_k=len(images))

# 플롯
fig, axes = plt.subplots(1, len(images) + 1, figsize=(15, 3))

# 쿼리
axes[0].text(0.5, 0.5, f"Query:\n'{query_text}'",
             ha='center', va='center', fontsize=12)
axes[0].axis('off')

# 결과
for i, (idx, dist) in enumerate(zip(indices, distances)):
    axes[i + 1].imshow(images[idx.item()])
    axes[i + 1].set_title(f"#{i+1}\nDist: {dist.item()}")
    axes[i + 1].axis('off')

plt.tight_layout()
plt.savefig('search_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ 결과 저장: search_result.png")
```

### Step 8: 해시 코드 분석

```python
from app.xor_search import HashCodeAnalyzer
import torch

# 해시 코드 생성
hash_codes = retriever.encode_images(images)
labels = torch.tensor([0, 1, 2])  # 카테고리 라벨

# 분포 분석
distribution = HashCodeAnalyzer.analyze_hash_distribution(hash_codes)

print("\n=== 해시 코드 품질 분석 ===")
print(f"비트 균형도: {distribution['mean_balance']:.4f} (1.0에 가까울수록 좋음)")
print(f"비트 분산: {distribution['mean_variance']:.4f} (높을수록 좋음)")
print(f"비트 엔트로피: {distribution['mean_entropy']:.4f} (높을수록 좋음)")

# 충돌률
collision_rate = HashCodeAnalyzer.calculate_collision_rate(hash_codes, labels)
print(f"충돌률: {collision_rate:.4f} (낮을수록 좋음)")
```

---

## 🎯 완전한 Colab 노트북 예제

아래는 한 번에 실행할 수 있는 완전한 예제입니다:

### Option A: 빠른 테스트 (학습 없이 추론만)

```python
# 1. 환경 설정
!git clone https://github.com/hyunlord/near_duplicate_deep_hashing.git
%cd near_duplicate_deep_hashing
!pip install -q pytorch-lightning transformers datasets albumentations

# 2. 사전 학습된 모델 다운로드 (있다면)
# !wget https://your-model-url/checkpoint.ckpt -O checkpoints_multimodal/last.ckpt

# 3. 추론 코드 실행
# (위의 Step 6-7 코드 실행)
```

### Option B: 전체 학습 + 추론

```python
# 1. 환경 설정
!git clone https://github.com/hyunlord/near_duplicate_deep_hashing.git
%cd near_duplicate_deep_hashing
!pip install -q pytorch-lightning transformers datasets albumentations matplotlib

# 2. 학습 실행 (30-60분 소요)
!python main_multimodal_colab.py

# 3. 학습 완료 후 추론
# (위의 Step 6-7 코드 실행)
```

---

## ⚙️ 설정 커스터마이징

학습 설정을 변경하려면 `main_multimodal_colab.py`를 수정하거나, 직접 코드로 실행하세요:

```python
# 커스텀 설정으로 학습
from app.dataset_multimodal import ImageTextDataModule
from app.module_multimodal import MultiModalHashingModel
import pytorch_lightning as pl

config = {
    'model_name': 'google/siglip-base-patch16-384',
    'bit_list': [32, 64],  # 더 적은 해시 길이 (메모리 절약)
    'hash_hidden_dim': 512,
    'learning_rate': 3e-5,
    'max_epochs': 10,  # 빠른 테스트용
    'batch_groups': 4,  # 더 작은 배치
    'images_per_group': 3,
    'train_size': 1000,  # 더 적은 데이터
    'val_size': 100,
    # ... 기타 설정
}

# 데이터 모듈
data_module = ImageTextDataModule(config)

# 모델
model = MultiModalHashingModel(config)

# Trainer
trainer = pl.Trainer(
    max_epochs=config['max_epochs'],
    accelerator='gpu',
    devices=1,
    precision='16-mixed'
)

# 학습
trainer.fit(model, data_module)
```

---

## 🐛 문제 해결

### 1. OOM (Out of Memory) 에러

```python
# main_multimodal_colab.py 수정
config['batch_groups'] = 3          # 5 → 3
config['images_per_group'] = 3      # 4 → 3
config['bit_list'] = [64]           # [16, 32, 64] → [64]
config['train_size'] = 1000         # 2000 → 1000
```

### 2. 데이터셋 로딩 실패

```python
# 인터넷 연결 확인
!ping -c 3 huggingface.co

# 수동 다운로드
from datasets import load_dataset
dataset = load_dataset('hyunlord/query_image_anchor_positive_large', cache_dir='./cache')
```

### 3. 학습 속도가 느림

```python
# Mixed precision 확인
config['precision'] = '16-mixed'

# num_workers 조정
config['num_workers'] = 0  # Colab에서는 0이 빠를 수 있음
```

### 4. 모델 체크포인트를 찾을 수 없음

```python
# 저장된 체크포인트 확인
!ls -lh checkpoints_multimodal/

# 최신 체크포인트 찾기
import glob
checkpoints = glob.glob('checkpoints_multimodal/*.ckpt')
latest_ckpt = max(checkpoints, key=os.path.getctime) if checkpoints else None
print(f"최신 체크포인트: {latest_ckpt}")
```

---

## 📊 예상 성능

**Colab T4 GPU 기준:**
- 학습 속도: ~2-3분/epoch
- 총 학습 시간: 30 epochs × 2.5분 ≈ 75분
- GPU 메모리: ~10GB
- 검색 속도: 1000개 이미지 DB에서 ~50ms

**성능 지표 (30 epochs 학습 후):**
- Vision Hash Accuracy: 0.7-0.9
- Text Hash Accuracy: 0.7-0.9
- Hash Agreement: 0.6-0.8
- I2T Recall@10: 0.6-0.8
- T2I Recall@10: 0.6-0.8

---

## 💡 다음 단계

1. **더 많은 데이터로 학습**
   - MS-COCO Captions
   - Flickr30k
   - Conceptual Captions

2. **하이퍼파라미터 튜닝**
   - Loss 가중치 조정
   - 학습률 최적화
   - 해시 길이 실험

3. **실제 애플리케이션에 적용**
   - 이미지 검색 엔진
   - 콘텐츠 추천 시스템
   - 중복 탐지 시스템

---

## 📚 추가 리소스

- [README_MULTIMODAL.md](README_MULTIMODAL.md): 상세 문서
- [example_colab_usage.py](example_colab_usage.py): 추가 예제
- [GitHub Issues](https://github.com/hyunlord/near_duplicate_deep_hashing/issues): 질문 및 버그 리포트

Happy Coding! 🚀
