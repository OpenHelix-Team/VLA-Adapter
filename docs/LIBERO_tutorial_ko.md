# VLA-Adapter LIBERO 튜토리얼 (한국어)

RTX 4090(24GB) 단일 GPU 기준으로 **LIBERO 벤치마크 성능 평가**와 **LoRA 학습**을 진행하는 단계별 가이드입니다.

> 평가(eval)만 할 거면 학습용 RLDS 데이터셋(~10GB)은 **불필요**합니다. 시뮬레이터 + 사전학습 Pro 체크포인트만 있으면 됩니다.
> 직접 학습까지 하려면 RLDS 데이터셋이 필요합니다(아래 "학습 가이드" 참고).

---

## 0. 공통 환경 설정

### 0-1. Conda 환경 + 의존성
```bash
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter

# CUDA 버전은 nvidia-smi에 맞게 (4090이면 cu121/cu122 휠 사용)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
cd VLA-Adapter
pip install -e .
```

### 0-2. LIBERO 시뮬레이터 설치
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt

# EGL 렌더링 에러(eglQueryString) 방지 — 필요 시
sudo apt-get update && sudo apt-get install -y \
  libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

### 0-3. VLM 백본 다운로드 (Prismatic-VLMs, Qwen2.5-0.5B)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --local-dir pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
```

---

## 1. 평가 가이드 (사전학습 체크포인트로 성능 측정)

### 1-1. 체크포인트 다운로드 (Pro 권장)
```bash
mkdir -p outputs
huggingface-cli download VLA-Adapter/LIBERO-Spatial-Pro --local-dir outputs/LIBERO-Spatial-Pro
huggingface-cli download VLA-Adapter/LIBERO-Object-Pro  --local-dir outputs/LIBERO-Object-Pro
huggingface-cli download VLA-Adapter/LIBERO-Goal-Pro    --local-dir outputs/LIBERO-Goal-Pro
huggingface-cli download VLA-Adapter/LIBERO-Long-Pro    --local-dir outputs/LIBERO-Long-Pro
```

### 1-2. 평가 실행
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
  --task_suite_name libero_spatial \
  --use_pro_version True
```

suite ↔ 체크포인트 ↔ `--task_suite_name` 매칭:

| 체크포인트 | `--task_suite_name` |
|---|---|
| LIBERO-Spatial-Pro | `libero_spatial` |
| LIBERO-Object-Pro  | `libero_object` |
| LIBERO-Goal-Pro    | `libero_goal` |
| LIBERO-Long-Pro    | `libero_10` |

- 기본값: **10 tasks × 50 episodes = 500 trials**. 빠른 동작 확인은 `--num_trials_per_task 5` 추가.
- 성공률 로그 → `experiments/logs/`, 롤아웃 영상 → `rollouts/`
- 원 논문(original) 버전을 쓰려면 `--use_pro_version False` + 해당 original 체크포인트 경로 지정.

### 기대 성능 (Pro, H100 기준)
| Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|
| 99.6 | 99.6 | 98.2 | 96.4 | 98.5 |

> README 명시: **H100이 아닌 GPU에서는 수치가 약간 다를 수 있습니다**(OpenVLA-OFT도 동일 현상 언급). 4090에서도 소폭 차이는 정상입니다.

---

## 2. 학습 가이드 (LoRA 파인튜닝, RTX 4090 24GB)

### 2-1. RLDS 데이터셋 다운로드
학습에는 RLDS 포맷 데이터셋이 필요합니다(`Spatial/Object/Goal/Long`, 총 ~10GB).

```bash
# git-lfs 필요
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

> ⚠️ **중요**: 다운로드한 폴더 이름에서 `modified_` 접두어를 제거해야 코드가 인식하는 경로와 맞습니다.
> 예: `modified_libero_spatial_no_noops` → `libero_spatial_no_noops`

최종 디렉터리 구조 (`data/libero/` 아래 배치):
```
data
└── libero
    ├── libero_spatial_no_noops/1.0.0   (json + 16 tfrecord)
    ├── libero_object_no_noops/1.0.0    (json + 32 tfrecord)
    ├── libero_goal_no_noops/1.0.0      (json + 16 tfrecord)
    └── libero_10_no_noops/1.0.0        (json + 32 tfrecord)
```

### 2-2. flash-attn 설치 (학습 필수)
```bash
pip install packaging ninja
ninja --version; echo $?   # exit code 0 이어야 함
pip install "flash-attn==2.5.5" --no-build-isolation
# 빌드가 어려우면 release 페이지에서 cuda/torch에 맞는 .whl을 받아 설치:
# https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5
```

### 2-3. 학습 실행 (24GB → batch_size 4, lora_rank 64 ≈ 약 20GB)
```bash
data_name=libero_spatial_no_noops
current_time=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name $data_name \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 200000 \
  --max_steps 200005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "$data_name" \
  --run_id_note VLA-Adapter--${data_name}--${current_time} \
  > logs/VLA-Adapter--${data_name}--${current_time}.log 2>&1 &
```

**파라미터 참고**
- `--dataset_name`: `libero_spatial_no_noops` / `libero_object_no_noops` / `libero_goal_no_noops` / `libero_10_no_noops` 중 택1
- `--batch_size` / `--lora_rank`: 24GB는 `4 / 64` 권장(≈20GB). VRAM 더 작으면 `--batch_size 1 --grad_accumulation_steps 8`(≈9.6GB)
- `--use_pro_version True`: **Pro 버전 강력 권장**(성능 향상, 학습 속도 거의 동일, Policy 207MB)
- 학습된 모델은 `outputs/` 폴더에 저장(체크포인트당 ~3GB → 충분한 공간 확보)
- 진행 로그는 `logs/` 폴더에서 확인
- CALVIN으로 학습하려면 `--data_root_dir data`(`/libero` 제거) + `--dataset_name calvin_abc`

### 2-4. 학습한 모델 평가
1번 "평가 가이드"와 동일하되, `--pretrained_checkpoint`를 직접 학습한 `outputs/<run_id>` 경로로 지정하면 됩니다.

---

## 부록: 자주 겪는 문제
- `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` → 0-2의 mesa/EGL 패키지 설치
- `image_aug`로 학습한 모델 평가 시 `center_crop==True` 필요(eval 기본값 True라 보통 자동 충족)
- 환경이 맞는지 확인하려면 저장소의 `our_envs.txt`와 패키지 버전 대조

참고: 논문 https://arxiv.org/abs/2509.09372 · 모델 https://huggingface.co/VLA-Adapter
