<div align="center">

## Demo

<video src="demo/sandwich_assembly_bimanual_demo.mp4" controls width="80%"></video>

<br>

**Sandwich Assembly** &mdash; Pick bread from rack &rarr; place on tray &rarr; add lettuce &rarr; add ham &rarr; cover with bread

<sub>Cobot Magic &nbsp;|&nbsp; Bimanual 14-DOF &nbsp;|&nbsp; 3-Camera &nbsp;|&nbsp; 2&times; Speed</sub>

</div>

---

# ALOHA Real-World

Training, deployment, and evaluation pipeline for the ALOHA bimanual robot. For base installation, see the project root [`README.md`](../../../README.md). Successfully verified on [Cobot Magic](https://global.agilex.ai/products/cobot-magic).

## Directory Structure

```
experiments/robot/aloha/
├── train_files/
│   ├── train_aloha.sh              # Training launcher (4-GPU torchrun)
│   ├── setup_training.sh           # Dataset registration + optional local model loading
│   ├── download_models.sh          # Download pretrained models (Qwen, DINOv2, SigLIP, Prismatic VLM)
│   ├── qwen25.py                   # Drop-in replacement for local Qwen loading
│   ├── materialize_local_vision.py # Drop-in replacement for local vision model loading
│   └── dinosiglip_vit_local_vision.py  # Drop-in replacement for local DINOv2+SigLIP loading
├── eval_files/
│   ├── deploy_server.sh            # Launch inference server (MsgPack HTTP)
│   ├── run_eval_client.sh          # Real robot client (requires ROS)
│   └── run_eval_client_fake.sh     # Fake-data client (no ROS needed, for sanity-checking the pipeline)
├── run_cobot_client.py             # Real ROS inference loop (3 cameras + bimanual 14-DOF)
├── run_fake_cobot_client.py        # Fake-data inference loop (generates synthetic observations)
├── requirements_aloha.txt          # ALOHA-specific dependencies
└── README.md
```

## Prerequisites

After completing the base installation from the root directory, install the ALOHA-specific dependencies:

```bash
pip install -r experiments/robot/aloha/requirements_aloha.txt
```

> **Real-robot deployment only**: `run_cobot_client.py` depends on ROS (`rospy`, `cv_bridge`, `sensor_msgs`, etc.) and must be run inside a ROS environment on the robot machine. The fake client `run_fake_cobot_client.py` does not require ROS.

## Sanity Check: Verify the Inference Pipeline

To confirm the inference pipeline works end-to-end without training, download an example checkpoint and run the server + fake client.

```bash
# 1. Download the example checkpoint
huggingface-cli download --resume-download SII-CDZ/test_aloha_adapter \
  --local-dir /path/to/SII-CDZ/test_aloha_adapter

# 2. Start the inference server (set PRETRAINED_CHECKPOINT in deploy_server.sh to the path above)
bash experiments/robot/aloha/eval_files/deploy_server.sh

# 3. In another terminal, run the fake-data client (no ROS / real robot required)
bash experiments/robot/aloha/eval_files/run_eval_client_fake.sh
```

The fake client sends synthetic 480x640x3 images and 14-DOF joint states to the server and checks whether action sequences are returned correctly. See [Deployment](#deployment) for detailed configuration.

## Pipeline

```bash
# 0. (Optional) Download pretrained models locally
bash experiments/robot/aloha/train_files/download_models.sh

# 1. Convert hdf5 real-robot data to TFDS format
#    See: https://github.com/cheng-haha/rlds_sim/tree/main/aloha_realworld

# 2. Register the dataset
bash experiments/robot/aloha/train_files/setup_training.sh <dataset_name>

# 3. Train
bash experiments/robot/aloha/train_files/train_aloha.sh

# 4. Launch the inference server
bash experiments/robot/aloha/eval_files/deploy_server.sh

# 5. Run client-side evaluation
bash experiments/robot/aloha/eval_files/run_eval_client_fake.sh   # fake-data sanity check
bash experiments/robot/aloha/eval_files/run_eval_client.sh         # real-robot evaluation
```

## Training

<details>
<summary><b>Local Model Download (Optional)</b></summary>

If you cannot access the HF Hub or prefer fully offline training, download all pretrained models in advance:

```bash
bash experiments/robot/aloha/train_files/download_models.sh
# Or specify an HF token for private repos
HF_TOKEN=hf_xxx bash experiments/robot/aloha/train_files/download_models.sh
```

Models downloaded:

| Model | Local Path |
|-------|------------|
| `timm/vit_large_patch14_reg4_dinov2.lvd142m` | `${ROOT_DIR}/ai_models/timm/...` |
| `timm/ViT-SO400M-14-SigLIP` | `${ROOT_DIR}/ai_models/timm/...` |
| `Qwen/Qwen2.5-0.5B` | `${ROOT_DIR}/ai_models/Qwen/Qwen2.5-0.5B` |
| `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` | `${ROOT_DIR}/ai_models/Stanford-ILIAD/...` |

After using the `--local-models` flag, `setup_training.sh` patches the project source files (`qwen25.py`, `materialize.py`, `dinosiglip_vit.py`) with local paths. To revert to online loading:

```bash
cd <project_root>
git restore prismatic/models/backbones/llm/qwen25.py
git restore prismatic/models/materialize.py
git restore prismatic/models/backbones/vision/dinosiglip_vit.py
```

</details>

### Data Preparation

ALOHA data must be converted to TFDS format. See [rlds_sim/aloha_realworld](https://github.com/cheng-haha/rlds_sim/tree/main/aloha_realworld) for the conversion tool.

### Configuration

Before training, update the default path variables in the following scripts.

**Key variables in `train_aloha.sh`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_DIR` | `/path/to/root` | Storage root directory |
| `DATA_ROOT_DIR` | `${ROOT_DIR}/datasets/cobot_aloha/tfds` | TFDS data directory |
| `VLM_PATH` | `${ROOT_DIR}/ai_models/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` | VLM weights path |
| `DATASET_NAME` | `bowl_stack_and_shelf_aloha_realworld_50` | Dataset name |
| `WANDB_ENTITY` | `your-wandb-entity` | W&B user / team |
| `WANDB_PROJECT` | `vla_adapter` | W&B project name |

**Key variables in `setup_training.sh`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_DIR` | `/path/to/root` | Storage root directory |
| `LOCAL_QWEN_PATH` | `${ROOT_DIR}/ai_models/Qwen/Qwen2.5-0.5B` | Local Qwen model path |
| `LOCAL_TIMM_PATH` | `${ROOT_DIR}/ai_models/timm` | Local timm vision model path |

**Default training hyperparameters (built into `train_aloha.sh`):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 12 | Per-GPU batch size |
| `nproc-per-node` | 4 | Number of GPUs |
| `learning_rate` | 2e-4 | Learning rate |
| `max_steps` | 10005 | Maximum training steps |
| `lora_rank` | 64 | LoRA rank |
| `num_images_in_input` | 3 | Number of input images (front + left wrist + right wrist) |
| `use_pro_version` | True | Use the Pro version (recommended) |
| `use_minivlm` | True | Use MiniVLM |
| `image_aug` | True | Image augmentation |

> To adjust hyperparameters (e.g., fewer GPUs or a different batch size), edit the corresponding variables directly in `train_aloha.sh`.

### Dataset Registration

`setup_training.sh` automatically registers a new dataset entry in `configs.py`, `mixtures.py`, and `transforms.py` (ALOHA bimanual config: 3 image observation keys + bimanual joint encoding).

```bash
# Register dataset only (models loaded from HF Hub)
bash experiments/robot/aloha/train_files/setup_training.sh bowl_stack_and_shelf_aloha_realworld_50

# Register dataset + enable local model loading (offline environment)
bash experiments/robot/aloha/train_files/setup_training.sh bowl_stack_and_shelf_aloha_realworld_50 --local-models
```

### Launch Training

```bash
bash experiments/robot/aloha/train_files/train_aloha.sh
```

Training outputs are saved to `outputs/<dataset_name>/<MODE>-<timestamp>/`; logs go to the `logs/` directory. W&B runs in offline mode by default.

## Deployment

Deployment follows a **server–client** architecture: the server loads a checkpoint and exposes an inference API over MsgPack HTTP; the client collects observations and sends requests to obtain action sequences.

### Server

Update the following variables in `deploy_server.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PRETRAINED_CHECKPOINT` | `/path/to/checkpoint_dir` | Trained checkpoint directory |
| `PORT` | `8888` | Server port |
| `DEVICE` | `0` | GPU device ID |
| `MODEL_FAMILY` | `openvla` | Model family |

```bash
bash experiments/robot/aloha/eval_files/deploy_server.sh
```

### Fake-Data Client (No ROS Required)

Verifies the inference pipeline using synthetic observations — no real robot or ROS environment needed. Generates 480x640x3 fake images and 14-DOF fake joint states.

Update the following variables in `run_eval_client_fake.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLA_SERVER_URL` | `http://127.0.0.1:8888` | Server address |
| `TASK_LABEL` | `Use the right arm to stack...` | Task instruction |
| `UNNORM_KEY` | `bowl_stack_and_shelf_aloha_realworld_50` | Action un-normalization key (must match the training dataset) |

```bash
bash experiments/robot/aloha/eval_files/run_eval_client_fake.sh
```

### Real-Robot Client (Requires ROS)

Subscribes to 3 camera topics and bimanual joint states via ROS, queries the server for a new action sequence every `num_open_loop_steps` (default 25) steps, and supports multi-trial evaluation with automatic success-rate tracking.

**Default ROS topics:**

| Topic | Type | Description |
|-------|------|-------------|
| `/camera_f/color/image_raw` | Image | Front camera |
| `/camera_l/color/image_raw` | Image | Left wrist camera |
| `/camera_r/color/image_raw` | Image | Right wrist camera |
| `/puppet/joint_left` | JointState | Left arm joint state |
| `/puppet/joint_right` | JointState | Right arm joint state |
| `/master/joint_left` | JointState | Left arm joint command (published) |
| `/master/joint_right` | JointState | Right arm joint command (published) |

```bash
bash experiments/robot/aloha/eval_files/run_eval_client.sh
```

> To adapt to different robot hardware, modify the `OpenVLAConfig` dataclass in [`run_cobot_client.py`](./run_cobot_client.py) to adjust topic names, control frequency, open-loop steps, and other parameters.

## Notes

- Training defaults to 4 GPUs (`nproc-per-node 4`). For single-GPU training, update that value in `train_aloha.sh` and adjust `batch_size` accordingly.
- During real-robot evaluation, the operator is prompted to press Enter to start each trial; pressing Space + Enter stops the current trial early.
- Trial results (JSON) are saved under `experiments/logs/<unnorm_key>/`.
- For more ALOHA evaluation references, see [openvla-oft/experiments/robot/aloha](https://github.com/moojink/openvla-oft/tree/main/experiments/robot/aloha).
