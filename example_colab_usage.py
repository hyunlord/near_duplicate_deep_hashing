"""
Colab 환경에서 멀티모달 해싱 모델 사용 예시

이 파일은 학습된 모델로 이미지-텍스트 검색을 수행하는 예제입니다.
"""

import torch
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

from app.module_multimodal import MultiModalHashingModel
from app.xor_search import MultiModalRetrieval, BinaryHashXOR, HashCodeAnalyzer


# ===== 1. 모델 로드 =====
def load_trained_model(checkpoint_path='./checkpoints_multimodal/last.ckpt'):
    """학습된 모델 로드"""
    print(f"모델 로드 중: {checkpoint_path}")

    model = MultiModalHashingModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    return model, tokenizer, device


# ===== 2. 검색 시스템 초기화 =====
def initialize_retrieval_system(model, tokenizer, device, bit_length=64):
    """검색 시스템 초기화"""
    retriever = MultiModalRetrieval(
        model=model,
        tokenizer=tokenizer,
        bit_length=bit_length,
        device=device
    )

    print(f"검색 시스템 초기화 완료 (해시 길이: {bit_length} bits)")
    return retriever


# ===== 3. 이미지-텍스트 데이터베이스 준비 =====
def prepare_database(image_paths, captions):
    """
    이미지-텍스트 데이터베이스 준비

    Args:
        image_paths: List of image file paths
        captions: List of text captions

    Returns:
        images: List of PIL Images
        texts: List of strings
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        images.append(img)

    return images, captions


# ===== 4. 텍스트로 이미지 검색 (Text-to-Image) =====
def search_image_by_text_example(retriever, image_database, query_text, top_k=5):
    """
    텍스트 쿼리로 이미지 검색

    Args:
        retriever: MultiModalRetrieval 객체
        image_database: 이미지 데이터베이스 (리스트 또는 텐서)
        query_text: 검색할 텍스트
        top_k: 반환할 상위 결과 수
    """
    print(f"\n{'='*80}")
    print(f"텍스트로 이미지 검색: '{query_text}'")
    print(f"{'='*80}")

    # 검색
    indices, distances = retriever.search_image_by_text(
        query_text=query_text,
        image_database=image_database,
        top_k=top_k
    )

    print(f"\nTop-{top_k} 결과:")
    for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"  {rank}. 이미지 인덱스: {idx.item()}, Hamming 거리: {dist.item()}")

    return indices, distances


# ===== 5. 이미지로 텍스트 검색 (Image-to-Text) =====
def search_text_by_image_example(retriever, text_database, query_image_path, top_k=5):
    """
    이미지 쿼리로 텍스트 검색

    Args:
        retriever: MultiModalRetrieval 객체
        text_database: 텍스트 데이터베이스 (리스트)
        query_image_path: 쿼리 이미지 경로
        top_k: 반환할 상위 결과 수
    """
    print(f"\n{'='*80}")
    print(f"이미지로 텍스트 검색: {query_image_path}")
    print(f"{'='*80}")

    # 이미지 로드
    query_image = Image.open(query_image_path).convert('RGB')

    # 검색
    indices, distances = retriever.search_text_by_image(
        query_image=query_image,
        text_database=text_database,
        top_k=top_k
    )

    print(f"\nTop-{top_k} 결과:")
    for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"  {rank}. '{text_database[idx.item()]}' (Hamming 거리: {dist.item()})")

    return indices, distances


# ===== 6. 배치 검색 (효율적) =====
def batch_search_example(retriever, query_texts, image_database, top_k=5):
    """
    배치 텍스트로 이미지 검색 (여러 쿼리 동시 처리)

    Args:
        retriever: MultiModalRetrieval 객체
        query_texts: 검색할 텍스트 리스트
        image_database: 이미지 데이터베이스
        top_k: 반환할 상위 결과 수
    """
    print(f"\n{'='*80}")
    print(f"배치 검색: {len(query_texts)}개 쿼리")
    print(f"{'='*80}")

    # 배치 검색
    indices, distances = retriever.batch_search_t2i(
        query_texts=query_texts,
        image_database=image_database,
        top_k=top_k
    )

    print(f"\n결과 shape: {indices.shape}")

    for i, query_text in enumerate(query_texts):
        print(f"\nQuery {i+1}: '{query_text}'")
        print(f"  Top-{top_k} 인덱스: {indices[i].tolist()}")
        print(f"  Hamming 거리: {distances[i].tolist()}")

    return indices, distances


# ===== 7. 해시 코드 데이터베이스 사전 계산 및 저장 =====
def precompute_and_save_database(retriever, images, texts, save_path='hash_db.pt'):
    """
    해시 코드 DB를 사전 계산하여 저장 (재사용 가능)

    Args:
        retriever: MultiModalRetrieval 객체
        images: 이미지 리스트
        texts: 텍스트 리스트
        save_path: 저장 경로
    """
    print(f"\n{'='*80}")
    print("해시 코드 데이터베이스 사전 계산 중...")
    print(f"{'='*80}")

    retriever.save_hash_database(
        images=images,
        texts=texts,
        save_path=save_path
    )

    print(f"✓ 저장 완료: {save_path}")

    # 로드 테스트
    db = retriever.load_hash_database(save_path)
    print(f"\n로드된 DB:")
    if 'image_codes' in db:
        print(f"  - Image codes: {db['image_codes'].shape}")
    if 'text_codes' in db:
        print(f"  - Text codes: {db['text_codes'].shape}")

    return db


# ===== 8. 사전 계산된 DB로 빠른 검색 =====
def fast_search_with_precomputed(retriever, query_text, precomputed_db, top_k=5):
    """
    사전 계산된 해시 DB로 빠른 검색

    Args:
        retriever: MultiModalRetrieval 객체
        query_text: 검색 텍스트
        precomputed_db: 사전 계산된 해시 코드 DB
        top_k: 반환할 상위 결과 수
    """
    print(f"\n{'='*80}")
    print(f"사전 계산 DB로 빠른 검색: '{query_text}'")
    print(f"{'='*80}")

    # 쿼리 인코딩
    query_code = retriever.encode_texts([query_text])

    # 빠른 검색
    indices, distances = retriever.fast_search_with_precomputed(
        query_code=query_code,
        precomputed_db_codes=precomputed_db['image_codes'],
        top_k=top_k
    )

    print(f"\nTop-{top_k} 결과:")
    for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
        print(f"  {rank}. 인덱스: {idx.item()}, Hamming 거리: {dist.item()}")

    return indices, distances


# ===== 9. 해시 코드 품질 분석 =====
def analyze_hash_quality(retriever, images, labels):
    """
    생성된 해시 코드의 품질 분석

    Args:
        retriever: MultiModalRetrieval 객체
        images: 이미지 리스트
        labels: 라벨 텐서
    """
    print(f"\n{'='*80}")
    print("해시 코드 품질 분석")
    print(f"{'='*80}")

    # 해시 코드 생성
    hash_codes = retriever.encode_images(images)

    # 분포 분석
    distribution = HashCodeAnalyzer.analyze_hash_distribution(hash_codes)

    print(f"\n비트 균형도 (평균): {distribution['mean_balance']:.4f} (1.0에 가까울수록 좋음)")
    print(f"비트 분산 (평균): {distribution['mean_variance']:.4f} (높을수록 좋음)")
    print(f"비트 엔트로피 (평균): {distribution['mean_entropy']:.4f} (높을수록 좋음)")

    # 충돌률 분석
    collision_rate = HashCodeAnalyzer.calculate_collision_rate(hash_codes, labels)
    print(f"\n충돌률: {collision_rate:.4f} (낮을수록 좋음)")

    # 시각화
    HashCodeAnalyzer.visualize_hamming_distance_distribution(
        hash_codes, labels, save_path='hamming_dist_plot.png'
    )

    return distribution, collision_rate


# ===== 10. 시각화: 검색 결과 표시 =====
def visualize_search_results(query, results_images, distances, is_text_query=True):
    """
    검색 결과 시각화

    Args:
        query: 쿼리 (텍스트 or 이미지)
        results_images: 검색 결과 이미지 리스트
        distances: Hamming 거리 리스트
        is_text_query: 텍스트 쿼리 여부
    """
    n_results = len(results_images)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 3))

    # 쿼리 표시
    if is_text_query:
        axes[0].text(0.5, 0.5, f"Query:\n'{query}'",
                    ha='center', va='center', fontsize=10, wrap=True)
        axes[0].axis('off')
    else:
        axes[0].imshow(query)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

    # 결과 표시
    for i, (img, dist) in enumerate(zip(results_images, distances)):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"#{i+1}\nDist: {dist}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('search_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ 결과 시각화 저장: search_results.png")


# ===== 메인 실행 예시 =====
def main():
    """전체 워크플로우 예시"""

    # 1. 모델 로드
    model, tokenizer, device = load_trained_model('./checkpoints_multimodal/last.ckpt')

    # 2. 검색 시스템 초기화
    retriever = initialize_retrieval_system(model, tokenizer, device, bit_length=64)

    # 3. 데이터베이스 준비 (예시)
    # image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', ...]
    # captions = ['a cat', 'a dog', 'a bird', ...]
    # images, texts = prepare_database(image_paths, captions)

    # 4. 텍스트로 이미지 검색
    # indices, distances = search_image_by_text_example(
    #     retriever, images, query_text="a cat sitting", top_k=5
    # )

    # 5. 이미지로 텍스트 검색
    # indices, distances = search_text_by_image_example(
    #     retriever, texts, query_image_path="query.jpg", top_k=5
    # )

    # 6. 배치 검색
    # query_texts = ["a cat", "a dog", "a bird"]
    # batch_search_example(retriever, query_texts, images, top_k=5)

    # 7. 해시 DB 사전 계산 및 저장
    # db = precompute_and_save_database(retriever, images, texts, 'hash_db.pt')

    # 8. 빠른 검색
    # fast_search_with_precomputed(retriever, "a cat", db, top_k=5)

    # 9. 해시 품질 분석
    # labels = torch.tensor([0, 0, 1, 1, 2, 2, ...])  # 라벨
    # analyze_hash_quality(retriever, images, labels)

    print("\n✓ 예제 코드 준비 완료!")
    print("  위 주석을 해제하고 실제 데이터로 실행하세요.")

    return retriever


if __name__ == '__main__':
    main()
