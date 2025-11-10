"""
Colab 올인원 실행 스크립트

이 파일을 Colab에서 순서대로 실행하면 됩니다.
각 섹션을 개별 셀로 복사해서 실행하세요.
"""

# ================================================================================
# STEP 1: 환경 설정 및 패키지 설치
# ================================================================================
"""
# 새 Colab 셀에서 실행

# 저장소 클론 (처음 한 번만)
!git clone https://github.com/hyunlord/near_duplicate_deep_hashing.git
%cd near_duplicate_deep_hashing

# 최신 코드 받기 (이미 클론한 경우)
!git pull origin claude/analyze-codebase-011CUr4if3NT1maqmXfPG2i2

# 패키지 설치
!pip install -q pytorch-lightning transformers datasets albumentations matplotlib

# GPU 확인
import torch
print("="*80)
print("GPU 확인")
print("="*80)
print(f"✓ GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ GPU를 사용할 수 없습니다. 런타임 > 런타임 유형 변경 > GPU를 선택하세요.")
print("="*80)
"""


# ================================================================================
# STEP 2: 학습 실행 (30-60분 소요)
# ================================================================================
"""
# 새 Colab 셀에서 실행

# 방법 1: 스크립트 직접 실행
!python main_multimodal_colab.py

# 방법 2: 설정 커스터마이징 (메모리가 부족한 경우)
# main_multimodal_colab.py 파일을 열어서 config를 수정하거나,
# 아래처럼 직접 코드로 실행
"""

# 커스텀 설정으로 학습 (메모리 부족 시)
def train_with_custom_config():
    from app.dataset_multimodal import ImageTextDataModule
    from app.module_multimodal import MultiModalHashingModel
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    import torch

    config = {
        # 모델
        'model_name': 'google/siglip-base-patch16-384',
        'bit_list': [64],  # 메모리 절약: 한 개만
        'hash_hidden_dim': 512,

        # 학습
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'margin': 0.5,
        'max_epochs': 10,  # 빠른 테스트용

        # Loss 가중치
        'lambda_ortho': 0.05,
        'lambda_lcs': 1.0,
        'lambda_cross': 1.0,
        'lambda_align': 0.5,
        'lambda_quant': 0.3,

        # 데이터 (메모리 절약)
        'dataset_name': 'hyunlord/query_image_anchor_positive_large',
        'train_size': 1000,  # 축소
        'val_size': 100,
        'test_size': 100,
        'batch_groups': 3,  # 축소
        'images_per_group': 3,  # 축소
        'image_size': 384,
        'text_max_length': 77,
        'num_workers': 0,  # Colab에서는 0이 안전

        # 학습 설정
        'accumulate_grad_batches': 4,  # 증가
        'precision': '16-mixed',

        # 경로
        'checkpoint_dir': './checkpoints_multimodal',
        'log_dir': './logs_multimodal',
    }

    print("="*80)
    print("커스텀 설정으로 학습 시작")
    print("="*80)
    print(f"배치 크기: {config['batch_groups']} × {config['images_per_group']} = {config['batch_groups'] * config['images_per_group']}")
    print(f"학습 데이터: {config['train_size']}")
    print(f"에포크: {config['max_epochs']}")
    print(f"해시 길이: {config['bit_list']}")
    print("="*80)

    # 데이터
    data_module = ImageTextDataModule(config)

    # 모델
    model = MultiModalHashingModel(config)

    # 콜백
    callbacks = [
        ModelCheckpoint(
            dirpath=config['checkpoint_dir'],
            filename='multimodal-{epoch:02d}',
            save_top_k=2,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=config['precision'],
        callbacks=callbacks,
        logger=CSVLogger(config['log_dir'], name='multimodal_hashing'),
        log_every_n_steps=10,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=1.0,
    )

    # 학습
    trainer.fit(model, data_module)

    print("\n✓ 학습 완료!")
    return model, trainer


# ================================================================================
# STEP 3: 학습 로그 확인
# ================================================================================
"""
# 새 Colab 셀에서 실행

# 최근 로그 확인
!tail -n 30 logs_multimodal/multimodal_hashing/version_0/metrics.csv

# 저장된 체크포인트 확인
!ls -lh checkpoints_multimodal/
"""


# ================================================================================
# STEP 4: 학습된 모델 로드
# ================================================================================
"""
# 새 Colab 셀에서 실행
"""

def load_model():
    from app.module_multimodal import MultiModalHashingModel
    from app.xor_search import MultiModalRetrieval
    from transformers import AutoTokenizer
    import torch

    # 체크포인트 경로
    checkpoint_path = './checkpoints_multimodal/last.ckpt'

    print("="*80)
    print("모델 로드 중...")
    print("="*80)

    # 모델 로드
    model = MultiModalHashingModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained('google/siglip-base-patch16-384')

    # 검색 시스템
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retriever = MultiModalRetrieval(
        model=model,
        tokenizer=tokenizer,
        bit_length=64,
        device=device
    )

    print(f"✓ 모델 로드 완료 (device: {device})")
    print("="*80)

    return model, tokenizer, retriever


# ================================================================================
# STEP 5: 샘플 데이터 준비
# ================================================================================
"""
# 새 Colab 셀에서 실행
"""

def prepare_sample_data():
    from PIL import Image
    import requests
    from io import BytesIO

    print("="*80)
    print("샘플 데이터 준비 중...")
    print("="*80)

    # 샘플 이미지 URL (Unsplash - 저작권 걱정 없음)
    image_urls = [
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",  # cat
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",  # dog
        "https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=400",  # nature
        "https://images.unsplash.com/photo-1551244072-5d12893278ab?w=400",  # bird
        "https://images.unsplash.com/photo-1535591273668-578e31182c4f?w=400",  # fish
    ]

    images = []
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            images.append(img)
            print(f"  ✓ 이미지 {i+1}/{len(image_urls)} 로드 완료")
        except Exception as e:
            print(f"  ✗ 이미지 {i+1} 로드 실패: {e}")

    # 샘플 텍스트
    text_database = [
        "a cute cat sitting",
        "a happy dog playing",
        "beautiful mountain landscape",
        "a colorful bird flying",
        "a tropical fish swimming"
    ]

    print(f"✓ 이미지 {len(images)}개, 텍스트 {len(text_database)}개 준비 완료")
    print("="*80)

    return images, text_database


# ================================================================================
# STEP 6: 텍스트로 이미지 검색
# ================================================================================
"""
# 새 Colab 셀에서 실행
"""

def test_text_to_image_search(retriever, images):
    print("\n" + "="*80)
    print("텍스트로 이미지 검색 테스트")
    print("="*80)

    # 쿼리 텍스트
    queries = [
        "a cat",
        "a dog playing",
        "mountain scenery"
    ]

    for query_text in queries:
        print(f"\nQuery: '{query_text}'")
        indices, distances = retriever.search_image_by_text(
            query_text=query_text,
            image_database=images,
            top_k=3
        )

        print(f"Top-3 결과:")
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            print(f"  {rank}. 이미지 #{idx.item()}, Hamming 거리: {dist.item()}")

    print("="*80)


# ================================================================================
# STEP 7: 이미지로 텍스트 검색
# ================================================================================
"""
# 새 Colab 셀에서 실행
"""

def test_image_to_text_search(retriever, images, text_database):
    print("\n" + "="*80)
    print("이미지로 텍스트 검색 테스트")
    print("="*80)

    # 첫 번째 이미지로 검색
    for img_idx in [0, 1, 2]:
        print(f"\nQuery: 이미지 #{img_idx}")
        indices, distances = retriever.search_text_by_image(
            query_image=images[img_idx],
            text_database=text_database,
            top_k=3
        )

        print(f"Top-3 결과:")
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            print(f"  {rank}. '{text_database[idx.item()]}' (Hamming 거리: {dist.item()})")

    print("="*80)


# ================================================================================
# STEP 8: 결과 시각화
# ================================================================================
"""
# 새 Colab 셀에서 실행
"""

def visualize_search_results(retriever, images, query_text):
    import matplotlib.pyplot as plt

    print("\n" + "="*80)
    print("검색 결과 시각화")
    print("="*80)

    # 검색
    indices, distances = retriever.search_image_by_text(
        query_text=query_text,
        image_database=images,
        top_k=len(images)
    )

    # 플롯
    fig, axes = plt.subplots(1, len(images) + 1, figsize=(15, 3))

    # 쿼리
    axes[0].text(0.5, 0.5, f"Query:\n'{query_text}'",
                ha='center', va='center', fontsize=12, wrap=True)
    axes[0].axis('off')
    axes[0].set_title("Query", fontsize=14, fontweight='bold')

    # 결과
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        axes[i + 1].imshow(images[idx.item()])
        axes[i + 1].set_title(f"Rank #{i+1}\nDist: {dist.item()}", fontsize=10)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('search_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✓ 결과 저장: search_result.png")
    print("="*80)


# ================================================================================
# STEP 9: 해시 코드 품질 분석
# ================================================================================
"""
# 새 Colab 셀에서 실행
"""

def analyze_hash_quality(retriever, images):
    from app.xor_search import HashCodeAnalyzer
    import torch

    print("\n" + "="*80)
    print("해시 코드 품질 분석")
    print("="*80)

    # 해시 코드 생성
    hash_codes = retriever.encode_images(images)
    labels = torch.tensor([0, 1, 2, 3, 4])  # 각 이미지는 다른 카테고리

    # 분포 분석
    distribution = HashCodeAnalyzer.analyze_hash_distribution(hash_codes)

    print(f"\n비트 균형도: {distribution['mean_balance']:.4f} (1.0에 가까울수록 좋음)")
    print(f"비트 분산: {distribution['mean_variance']:.4f} (높을수록 좋음)")
    print(f"비트 엔트로피: {distribution['mean_entropy']:.4f} (높을수록 좋음)")

    # 충돌률
    collision_rate = HashCodeAnalyzer.calculate_collision_rate(hash_codes, labels)
    print(f"충돌률: {collision_rate:.4f} (낮을수록 좋음)")

    # 시각화
    try:
        HashCodeAnalyzer.visualize_hamming_distance_distribution(
            hash_codes, labels, save_path='hamming_dist.png'
        )
        print("\n✓ Hamming distance 분포 저장: hamming_dist.png")
    except Exception as e:
        print(f"\n⚠ 시각화 실패: {e}")

    print("="*80)

    return distribution, collision_rate


# ================================================================================
# STEP 10: 전체 파이프라인 실행
# ================================================================================
"""
# 새 Colab 셀에서 실행 - 모든 단계를 한 번에 실행
"""

def run_complete_pipeline():
    """전체 파이프라인 실행"""

    print("\n" + "="*80)
    print("멀티모달 해싱 - 전체 파이프라인")
    print("="*80 + "\n")

    # 1. 모델 로드
    print("STEP 1: 모델 로드")
    model, tokenizer, retriever = load_model()

    # 2. 샘플 데이터 준비
    print("\nSTEP 2: 샘플 데이터 준비")
    images, text_database = prepare_sample_data()

    # 3. 텍스트로 이미지 검색
    print("\nSTEP 3: 텍스트 → 이미지 검색")
    test_text_to_image_search(retriever, images)

    # 4. 이미지로 텍스트 검색
    print("\nSTEP 4: 이미지 → 텍스트 검색")
    test_image_to_text_search(retriever, images, text_database)

    # 5. 시각화
    print("\nSTEP 5: 결과 시각화")
    visualize_search_results(retriever, images, query_text="a cat")

    # 6. 품질 분석
    print("\nSTEP 6: 해시 코드 품질 분석")
    distribution, collision_rate = analyze_hash_quality(retriever, images)

    print("\n" + "="*80)
    print("✓ 전체 파이프라인 완료!")
    print("="*80)

    return retriever, images, text_database


# ================================================================================
# 실행 가이드
# ================================================================================
"""
============================================================
Colab에서 실행하는 방법
============================================================

1. 새 Colab 노트북 생성
2. 런타임 > 런타임 유형 변경 > GPU 선택
3. 이 파일(colab_all_in_one.py)을 Colab에 업로드하거나,
   각 섹션을 개별 셀로 복사

실행 순서:
----------

셀 1: STEP 1 코드 실행 (환경 설정)
셀 2: STEP 2 코드 실행 (학습 - 30-60분 소요)
     또는 train_with_custom_config() 실행
셀 3: 학습 완료 후, run_complete_pipeline() 실행

빠른 테스트 (학습 없이):
-----------------------

사전 학습된 모델이 있다면:
1. STEP 1 실행
2. STEP 4부터 실행 (모델 로드)
3. run_complete_pipeline() 실행

============================================================
"""


if __name__ == '__main__':
    print(__doc__)
