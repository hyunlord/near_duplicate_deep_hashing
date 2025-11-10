"""
이미지-텍스트 멀티모달 해싱 학습 스크립트
Colab 단일 GPU 환경 최적화 버전
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from app.dataset_multimodal import ImageTextDataModule
from app.module_multimodal import MultiModalHashingModel


def main():
    """메인 학습 함수"""

    # ===== 설정 =====
    config = {
        # 모델 설정
        'model_name': 'google/siglip-base-patch16-384',
        'bit_list': [16, 32, 64],  # Colab 메모리를 고려하여 축소 (기존: [8, 16, 32, 48, 64, 128])
        'hash_hidden_dim': 512,

        # 학습 하이퍼파라미터
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'margin': 0.5,

        # Loss 가중치 (Single-modal)
        'lambda_ortho': 0.05,
        'lambda_lcs': 1.0,

        # Loss 가중치 (Cross-modal)
        'lambda_cross': 1.0,      # Cross-modal triplet loss
        'lambda_align': 0.5,      # Modality alignment loss
        'lambda_quant': 0.3,      # Cross-modal quantization loss

        # 데이터 설정
        'dataset_name': 'hyunlord/query_image_anchor_positive_large',
        'train_size': 2000,       # Colab 메모리 고려 (기존: 4000)
        'val_size': 200,          # 기존: 400
        'test_size': 200,
        'batch_groups': 5,        # 기존: 10
        'images_per_group': 4,    # 기존: 10
        'image_size': 384,
        'text_max_length': 77,
        'num_workers': 2,

        # 학습 설정
        'max_epochs': 30,         # Colab 시간 제약 고려 (기존: 50)
        'accumulate_grad_batches': 2,  # Gradient accumulation (메모리 절약)
        'precision': '16-mixed',  # Mixed precision (메모리 절약)

        # 체크포인트 설정
        'checkpoint_dir': './checkpoints_multimodal',
        'log_dir': './logs_multimodal',
    }

    print("=" * 80)
    print("멀티모달 해싱 학습 시작 (Colab 최적화 버전)")
    print("=" * 80)
    print(f"모델: {config['model_name']}")
    print(f"해시 길이: {config['bit_list']}")
    print(f"배치 크기: {config['batch_groups']} groups × {config['images_per_group']} images = {config['batch_groups'] * config['images_per_group']}")
    print(f"학습 데이터: {config['train_size']} samples")
    print(f"검증 데이터: {config['val_size']} samples")
    print(f"최대 에포크: {config['max_epochs']}")
    print(f"Precision: {config['precision']}")
    print(f"Gradient Accumulation: {config['accumulate_grad_batches']} steps")
    print("=" * 80)

    # GPU 확인
    if torch.cuda.is_available():
        print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ GPU를 사용할 수 없습니다. CPU로 학습합니다.")
    print("=" * 80)

    # ===== 데이터 모듈 =====
    print("\n데이터 로딩 중...")
    data_module = ImageTextDataModule(config)

    # ===== 모델 =====
    print("모델 초기화 중...")
    model = MultiModalHashingModel(config)

    # ===== 콜백 설정 =====
    callbacks = [
        # 체크포인트: 최고 성능 모델 저장
        ModelCheckpoint(
            dirpath=config['checkpoint_dir'],
            filename='multimodal-{epoch:02d}-{val/64_final_score:.4f}' if 64 in config['bit_list'] else 'multimodal-{epoch:02d}',
            monitor='val/64_final_score' if 64 in config['bit_list'] else 'train/total_loss',
            mode='max' if 64 in config['bit_list'] else 'min',
            save_top_k=3,
            save_last=True,
            verbose=True
        ),

        # 학습률 모니터링
        LearningRateMonitor(logging_interval='step'),

        # Early stopping (선택적)
        # EarlyStopping(
        #     monitor='val/64_final_score' if 64 in config['bit_list'] else 'train/total_loss',
        #     patience=10,
        #     mode='max' if 64 in config['bit_list'] else 'min',
        #     verbose=True
        # )
    ]

    # ===== 로거 =====
    csv_logger = CSVLogger(
        save_dir=config['log_dir'],
        name='multimodal_hashing'
    )

    # ===== Trainer 설정 (단일 GPU) =====
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,  # 단일 GPU
        precision=config['precision'],
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=10,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=1.0,  # Gradient clipping

        # 메모리 최적화
        enable_model_summary=True,

        # Colab 환경 고려
        deterministic=False,  # 속도 우선
        benchmark=True,       # cuDNN 벤치마크 활성화
    )

    # ===== 학습 시작 =====
    print("\n" + "=" * 80)
    print("학습 시작!")
    print("=" * 80 + "\n")

    try:
        trainer.fit(model, data_module)

        print("\n" + "=" * 80)
        print("학습 완료!")
        print("=" * 80)

        # 최고 성능 모델 경로
        best_model_path = callbacks[0].best_model_path
        print(f"\n최고 성능 모델 저장 위치: {best_model_path}")

        # 검증 수행
        if config['val_size'] > 0:
            print("\n검증 수행 중...")
            trainer.validate(model, data_module)

    except KeyboardInterrupt:
        print("\n\n학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        raise

    return model, trainer


def test_inference():
    """학습된 모델로 추론 테스트"""
    from app.xor_search import MultiModalRetrieval
    from transformers import AutoTokenizer
    from PIL import Image

    # 체크포인트 로드
    checkpoint_path = './checkpoints_multimodal/last.ckpt'

    if not os.path.exists(checkpoint_path):
        print(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return

    print(f"모델 로드 중: {checkpoint_path}")
    model = MultiModalHashingModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)

    # 검색 시스템 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retriever = MultiModalRetrieval(
        model=model,
        tokenizer=tokenizer,
        bit_length=64,  # 사용할 해시 길이
        device=device
    )

    print("\n추론 준비 완료!")
    print("=" * 80)

    # 예시: 텍스트로 이미지 검색
    # query_text = "a cat sitting on a chair"
    # indices, distances = retriever.search_image_by_text(query_text, image_database, top_k=10)
    # print(f"Query: {query_text}")
    # print(f"Top-10 indices: {indices}")
    # print(f"Hamming distances: {distances}")

    return retriever


if __name__ == '__main__':
    # 학습 실행
    model, trainer = main()

    # 학습 후 추론 테스트 (선택적)
    # retriever = test_inference()
