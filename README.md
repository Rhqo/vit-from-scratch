# Vision Transformer from Scratch

## 프로젝트 설명

이 프로젝트는 PyTorch를 사용하여 Vision Transformer (ViT) 모델을 처음부터 구현합니다. CIFAR-10 데이터셋을 사용하여 모델을 학습하고 평가합니다.

## 파일 설명

*   `train.py`: ViT 모델 학습 스크립트 \
    - CIFAR-10 데이터셋 다운로드 및 `DataLoader` 형태로 변환
    - `model/vit.py`에 정의된 ViT 모델을 initialize
    - 학습 과정에서의 loss와 accuracy를 출력
    - 학습이 완료된 모델을 `checkpoints` 디렉토리에 저장
*   `inference.py`: 학습된 ViT 모델을 사용하여 inference 진행
    - `checkpoints` 디렉토리에서 학습된 모델 load
    - CIFAR-10 test 데이터셋의 이미지 classification, 실제 레이블과 함께 시각화
*   `utils/download_dataset.py`: CIFAR-10 데이터셋을 다운로드 및 `DataLoader` 형태로 변환
    - 다운로드된 데이터셋의 위치: `data/cifar10-batches-py`
*   `model/vit.py`: ViT 모델 구현

## ViT 모델 구현

### 1. ViTConfig

```python
@dataclass
class ViTConfig:
    # Embedding
    num_channels: int = 3
    embed_dim: int = 256
    image_size: int = 32
    patch_size: int = 16
    # EncoderBlock
    num_attention_heads: int = 8
    attention_dropout: float = 0.0
    # Encoder
    num_encoder_blocks: int = 6
    # MLP
    mlp_hidden_dim: int = 256*2
    mlp_dropout: float = 0.0
    # LayerNorm
    layer_norm_eps: float = 1e-6
    # Training
    batch_size = 32
    epochs = 10
    learning_rate = 3e-4
    num_classes = 10
```

### PatchEmbedding

이미지를 패치로 나누고, 각 패치를 linear projection하여 임베딩 벡터를 생성하는 클래스입니다.

*   `proj`: `nn.Conv2d` 레이어를 사용하여 이미지를 패치로 나누고 임베딩을 수행합니다. `kernel_size`와 `stride`를 `patch_size`와 동일하게 설정하여 이미지를 겹치지 않는 패치로 나눕니다.
*   `cls_token`: 분류를 위해 사용되는 특별한 토큰입니다. `nn.Parameter`로 선언되어 학습 과정에서 업데이트됩니다.
*   `pos_embed`: 각 패치의 위치 정보를 담고 있는 포지셔널 임베딩입니다. `nn.Parameter`로 선언되어 학습 과정에서 업데이트됩니다.


### MLP (Multi-Layer Perceptron)

Transformer 인코더 블록 내부에 사용되는 MLP 레이어입니다.

### TransformerEncoderLayer

Transformer 인코더의 단일 블록을 구성하는 클래스입니다. Multi-Head Self-Attention 레이어와 MLP 레이어로 구성됩니다.

*   `norm1`, `norm2`: `nn.LayerNorm` 레이어입니다. 각각 MHSA와 MLP 레이어 이전에 적용됩니다.
*   `attn`: `nn.MultiheadAttention` 레이어입니다.
*   `mlp`: `MLP` 클래스의 인스턴스입니다.
