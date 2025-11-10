# 이미지-텍스트 멀티모달 해싱 (Colab 최적화)

이미지와 텍스트를 동일한 1-bit binary hash 공간에 매핑하여 XOR 연산으로 빠른 크로스 모달 검색을 수행하는 시스템입니다.

## 🎯 주요 특징

- **멀티모달**: 이미지와 텍스트를 동일한 해시 공간에 매핑
- **1-bit 해싱**: Binary hash (-1/+1) 사용
- **XOR 검색**: 초고속 Hamming distance 기반 검색
- **다중 해시 길이**: 8, 16, 32, 48, 64, 128 비트 동시 학습
- **Colab 최적화**: 단일 GPU 환경에 최적화된 메모리 효율적 구현

## 📋 요구사항

```bash
pip install torch torchvision
pip install pytorch-lightning
pip install transformers
pip install datasets
pip install albumentations
pip install matplotlib
```

## 🚀 빠른 시작

### 1. 학습

```python
# main_multimodal_colab.py 실행
python main_multimodal_colab.py
```

**Colab 최적화 설정:**
- 배치 크기: 5 groups × 4 images = 20
- Mixed precision: fp16
- Gradient accumulation: 2 steps
- 학습 데이터: 2000 samples (기존 4000)
- 해시 길이: [16, 32, 64] bits (메모리 절약)

### 2. 추론 및 검색

```python
from app.module_multimodal import MultiModalHashingModel
from app.xor_search import MultiModalRetrieval
from transformers import AutoTokenizer

# 모델 로드
model = MultiModalHashingModel.load_from_checkpoint('checkpoints_multimodal/last.ckpt')
tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)

# 검색 시스템 초기화
retriever = MultiModalRetrieval(
    model=model,
    tokenizer=tokenizer,
    bit_length=64,
    device='cuda'
)

# 텍스트로 이미지 검색
indices, distances = retriever.search_image_by_text(
    query_text="a cat sitting on a chair",
    image_database=images,  # (N, 3, 384, 384) or list of PIL Images
    top_k=10
)

# 이미지로 텍스트 검색
indices, distances = retriever.search_text_by_image(
    query_image=query_img,  # PIL Image or (3, 384, 384) tensor
    text_database=texts,    # List of strings
    top_k=10
)
```

## 📁 파일 구조

```
.
├── app/
│   ├── dataset_multimodal.py      # 이미지-텍스트 데이터 로더 (단일 GPU)
│   ├── module_multimodal.py       # 멀티모달 해싱 모델
│   └── xor_search.py              # XOR 기반 검색 시스템
├── main_multimodal_colab.py       # Colab 학습 스크립트
├── example_colab_usage.py         # 사용 예제 모음
└── README_MULTIMODAL.md           # 이 문서
```

## 🔧 주요 컴포넌트

### 1. MultiModalHashingModel

이미지와 텍스트를 동일한 해시 공간에 매핑하는 모델입니다.

**아키텍처:**
```
Image → Vision Encoder → Vision Hash Layer → Hash Codes
Text  → Text Encoder   → Text Hash Layer   → Hash Codes
```

**Loss 함수:**
- **Single-modal losses**: Triplet loss, Orthogonal loss, LCS loss
- **Cross-modal losses**:
  - Cross-modal triplet loss (이미지-텍스트 대조 학습)
  - Modality alignment loss (같은 라벨의 이미지-텍스트 정렬)
  - Quantization loss (이진화 후에도 일치 유지)

### 2. XOR 검색 시스템

**BinaryHashXOR 클래스:**
```python
# 연속 임베딩 → Binary hash (0/1)
binary_codes = BinaryHashXOR.sign_to_binary(embeddings)

# Hamming distance 계산 (XOR)
hamming_dist = BinaryHashXOR.hamming_distance(codes1, codes2)

# Top-K 검색
indices, distances = BinaryHashXOR.xor_retrieval(query_codes, db_codes, top_k=10)
```

**MultiModalRetrieval 클래스:**
- `encode_images()`: 이미지 → Binary hash
- `encode_texts()`: 텍스트 → Binary hash
- `search_image_by_text()`: 텍스트 쿼리로 이미지 검색
- `search_text_by_image()`: 이미지 쿼리로 텍스트 검색
- `batch_search_*()`: 배치 검색 (효율적)
- `save_hash_database()`: 해시 DB 사전 계산 및 저장
- `load_hash_database()`: 사전 계산된 DB 로드

### 3. 데이터셋

**SimpleBatchSampler:**
- 그룹 기반 샘플링 (triplet mining에 적합)
- 단일 GPU 최적화

**ImageTextDataModule:**
- 이미지-텍스트 페어 데이터 로더
- 토크나이저 통합
- Colab 메모리 고려한 작은 배치 크기

## 💡 사용 예제

### 예제 1: 기본 학습 및 검색

```python
# 1. 학습
python main_multimodal_colab.py

# 2. 모델 로드
model = MultiModalHashingModel.load_from_checkpoint('checkpoints_multimodal/last.ckpt')
tokenizer = AutoTokenizer.from_pretrained('google/siglip-base-patch16-384')

# 3. 검색 시스템 초기화
retriever = MultiModalRetrieval(model, tokenizer, bit_length=64, device='cuda')

# 4. 검색
indices, distances = retriever.search_image_by_text("a cat", images, top_k=5)
print(f"Top-5 indices: {indices}")
print(f"Hamming distances: {distances}")
```

### 예제 2: 배치 검색 (효율적)

```python
# 여러 텍스트 쿼리로 동시 검색
query_texts = ["a cat", "a dog", "a bird"]
indices, distances = retriever.batch_search_t2i(query_texts, images, top_k=10)

# 결과: (3, 10) shape - 각 쿼리당 top-10 결과
for i, query in enumerate(query_texts):
    print(f"Query: {query}")
    print(f"  Top-10 indices: {indices[i]}")
    print(f"  Distances: {distances[i]}")
```

### 예제 3: 해시 DB 사전 계산 (재사용)

```python
# 1. 해시 코드 DB 사전 계산 및 저장
retriever.save_hash_database(
    images=image_list,
    texts=text_list,
    save_path='hash_db.pt'
)

# 2. 사전 계산된 DB 로드
db = retriever.load_hash_database('hash_db.pt')

# 3. 빠른 검색 (인코딩 없이 바로 XOR)
query_code = retriever.encode_texts(["a cat"])
indices, distances = retriever.fast_search_with_precomputed(
    query_code, db['image_codes'], top_k=10
)
```

### 예제 4: 해시 코드 품질 분석

```python
from app.xor_search import HashCodeAnalyzer

# 해시 코드 생성
hash_codes = retriever.encode_images(images)
labels = torch.tensor([0, 0, 1, 1, 2, 2, ...])

# 분포 분석
distribution = HashCodeAnalyzer.analyze_hash_distribution(hash_codes)
print(f"균형도: {distribution['mean_balance']:.4f}")
print(f"분산: {distribution['mean_variance']:.4f}")
print(f"엔트로피: {distribution['mean_entropy']:.4f}")

# 충돌률 분석
collision_rate = HashCodeAnalyzer.calculate_collision_rate(hash_codes, labels)
print(f"충돌률: {collision_rate:.4f}")

# 시각화
HashCodeAnalyzer.visualize_hamming_distance_distribution(
    hash_codes, labels, save_path='hamming_dist.png'
)
```

## ⚙️ 설정 (config)

```python
config = {
    # 모델
    'model_name': 'google/siglip-base-patch16-384',
    'bit_list': [16, 32, 64],  # 해시 길이
    'hash_hidden_dim': 512,

    # 학습
    'learning_rate': 3e-5,
    'max_epochs': 30,
    'margin': 0.5,

    # Loss 가중치 (Single-modal)
    'lambda_ortho': 0.05,
    'lambda_lcs': 1.0,

    # Loss 가중치 (Cross-modal)
    'lambda_cross': 1.0,   # Cross-modal triplet
    'lambda_align': 0.5,   # Modality alignment
    'lambda_quant': 0.3,   # Quantization

    # 데이터
    'train_size': 2000,
    'batch_groups': 5,
    'images_per_group': 4,

    # Colab 최적화
    'accumulate_grad_batches': 2,
    'precision': '16-mixed',
}
```

## 📊 성능 메트릭

### 학습 메트릭
- `train/total_loss`: 전체 손실
- `train/{bit}_vision_loss`: Vision 손실
- `train/{bit}_text_loss`: Text 손실
- `train/{bit}_cross_loss`: Cross-modal 손실
- `train/lcs_loss`: Long-Short Cascade 손실

### 검증 메트릭
- `val/{bit}_vision_pos_acc`: Vision positive 정확도
- `val/{bit}_text_pos_acc`: Text positive 정확도
- `val/{bit}_cross_modal_sim`: 이미지-텍스트 유사도
- `val/{bit}_hash_agreement`: 해시 코드 일치율
- `val/{bit}_i2t_recall@10`: Image-to-Text Recall@10
- `val/{bit}_t2i_recall@10`: Text-to-Image Recall@10
- `val/{bit}_final_score`: 종합 점수

**Final Score 계산:**
```python
final_score = 0.25 * vision_pos_acc +
              0.25 * text_pos_acc +
              0.20 * hash_agreement +
              0.15 * i2t_recall +
              0.15 * t2i_recall
```

## 🔍 XOR 연산 원리

### 1-bit Binary Hash
```python
연속 임베딩: [0.5, -0.3, 0.8, -0.1, ...]
           ↓ torch.sign()
해시 코드:   [+1,  -1,  +1,  -1, ...]  # -1/+1
           ↓ (sign + 1) / 2
Binary:     [1,   0,   1,   0, ...]  # 0/1
```

### XOR 거리 계산
```python
이미지 비트: [1, 0, 1, 0, 1, 1, 0, 0]
텍스트 비트: [1, 0, 1, 1, 1, 0, 0, 1]
            ↓ XOR
결과:        [0, 0, 0, 1, 0, 1, 0, 1]  # 다른 비트만 1
            ↓ sum
Hamming 거리: 3 (64개 비트 중 3개만 다름 → 매우 유사)
```

**장점:**
- ⚡ 초고속: 비트 연산으로 CPU에서도 매우 빠름
- 💾 압축적: 64-bit = 8 bytes (매우 작음)
- 🚀 확장성: 수백만 이미지도 실시간 검색 가능
- 🔧 하드웨어 최적화: GPU/CPU 모두 지원

## 🎓 알고리즘 상세

### Hard Triplet Mining
```python
# 배치 내에서 가장 어려운 triplet 선택
- Anchor: 기준 샘플
- Hard Positive: 같은 라벨 중 가장 먼 샘플
- Hard Negative: 다른 라벨 중 가장 가까운 샘플

→ 모델이 어려운 케이스를 집중 학습
```

### Long-Short Cascade (LCS)
```python
# 긴 해시 코드가 짧은 해시 코드를 가르침
128-bit → 64-bit
64-bit  → 32-bit
32-bit  → 16-bit

→ 지식 증류로 모든 해시 길이의 성능 향상
```

### Cross-Modal Alignment
```python
# 같은 라벨의 이미지-텍스트를 가깝게 매핑
image_embed ≈ text_embed (같은 라벨)
image_embed ≠ text_embed (다른 라벨)

→ 크로스 모달 검색 성능 향상
```

## 🐛 문제 해결

### OOM (Out of Memory) 에러
```python
# config에서 다음을 줄이세요:
config['batch_groups'] = 4          # 5 → 4
config['images_per_group'] = 3      # 4 → 3
config['bit_list'] = [32, 64]       # [16, 32, 64] → [32, 64]
config['train_size'] = 1000         # 2000 → 1000
```

### 학습 속도가 느림
```python
# Mixed precision 확인
config['precision'] = '16-mixed'

# Gradient accumulation 증가
config['accumulate_grad_batches'] = 4  # 2 → 4

# num_workers 조정
config['num_workers'] = 0  # Colab에서는 0이 빠를 수 있음
```

### 검색 결과가 부정확함
```python
# 더 긴 학습
config['max_epochs'] = 50

# Loss 가중치 조정
config['lambda_cross'] = 2.0   # Cross-modal loss 증가
config['lambda_align'] = 1.0   # Alignment loss 증가

# 더 긴 해시 사용
retriever = MultiModalRetrieval(..., bit_length=128)
```

## 📚 참고 자료

- **SigLIP**: Sigmoid Loss for Language Image Pre-Training
- **Deep Hashing**: Learning to Hash for Large-Scale Image Retrieval
- **Cross-Modal Retrieval**: Learning Cross-Modal Embeddings with Adversarial Networks

## 📝 라이선스

MIT License

## 🤝 기여

Pull requests are welcome!

## 📧 문의

Issues를 통해 질문해주세요.
