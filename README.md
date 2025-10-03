<div align="center">
  <img src="figure/LOGO2.png" width="70%" style="vertical-align:-7px;" />


[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2509.09372) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/VLA-Adapter) [![Twitter](https://img.shields.io/badge/AK-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/_akhaliq/status/1966610780838621241) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://github.com/OpenHelix-Team/VLA-Adapter/issues/1)

</div>

### The official implementation of **VLA-Adapter**.
<br/>

<div id="top" align="center">
<p align="center">
<img src=figure/Framework.png width=90% />
</p>
</div>

> **📝 Paper: https://arxiv.org/abs/2509.09372**<br/>
> **🌍 Project page: https://vla-adapter.github.io/**<br/>
> **🤗 HuggingFace: https://huggingface.co/VLA-Adapter**<br/>
> **Github: https://github.com/OpenHelix-Team/VLA-Adapter**

<br/>

## :loudspeaker: News!
- **[2025/09/22]** We released our codes! An enhanced **Pro** version is also released (this version conforms to the pipeline in the original paper, but is optimized in implementation). Everyone is welcome to use it!🎉
- **[2025/09/13]** Our paper won the 🥇**first place** in the [daily list](https://huggingface.co/papers/date/2025-09-12), the 🥈**second place** in the [weekly list](https://huggingface.co/papers/week/2025-W37), and 🥉**third place** in the [Monthly list](https://huggingface.co/papers/month/2025-09) in HF! ⭐
- **[2025/09/12]** We released the original version of the VLA-Adapter for four LIBERO models on [HuggingFace](https://huggingface.co/VLA-Adapter).
- **[2025/09/11]** We released our paper on [ArXiv](https://arxiv.org/abs/2509.09372).

<br/>

## :black_nib: TODO List<a name="todo"></a>

- [x]  Release **checkpoints** for reproduction.
- [x]  Release [VLA-Adapter v2 paper](https://arxiv.org/abs/2509.09372).
- [ ]  A more **powerful version**, **VLA-Adapter++**, and a detailed **technical report** 📝 will be released soon.<br/>
- [ ]  Continue to update the code to adapt to various **real-world systems** deployments, including the configuration of our paper, Franka, UR-5, and AGILE Piper.<br/>
- [ ]  It will soon be compatible with **various foundation models**, including but not limited to [VPP](https://arxiv.org/abs/2412.14803), [π0.5](https://arxiv.org/abs/2504.16054).<br/>
- [ ]  We will update the **diffusion transformers** and **flow matching** policy networks in the future, and the results will be updated in the subsequent VLA-Adapter++ technical report.
- [ ]  We will also update and give more experiments on **Frozen backbone**.
- [ ]  We will expand its **generalization** further in the future. Work is in progress! So please stay tuned!
- [ ]  **RL post-training** is also in progress. Interested researchers are welcome to join us in building this foundation!
- [ ]  **The dual-system compatibility** of VLA-Adapter is under exploration!


<br/>

## 🌟 Table of Contents

- [:rocket: Quick Start](#rocket-quick-start) 
  - [Conda Environment of VLA-Adapter](#conda-environment-of-vla-adapter)
  - [Install Dependencies](#install-dependencies)
- [:pencil: Data Preparation](#pencil-data-preparation) 
  - [LIBERO Benchmark](#libero-benchmark)
  - [CALVIN Benchmark](#calvin-benchmark)
  - [:video_game: Our Dependencies](#video_game-our-dependencies)
  - [:pushpin: Benchmark Location](#pushpin-benchmark-location)
- [⚓ VLM backbone](#vlm)
- [:fire: Training for Different Configurations](#fire-training-for-different-configurations) &emsp; => Provides **training configurations** for GPUs ranging from **10GB** to **80GB** of VRAM.
  - [:books: Related File for Training](#books-related-file-for-training)
  - [:ledger: How to Train on Extremely Limited VRAM GPUs](#ledger-how-to-train-on-extremely-limited-vram-gpus) &emsp; => A card with 10GB-12GB *(e.g. NVIDIA GeForce RTX 2080Ti, 3060, 3080, 4070, 4080, and 5070)*
  - [:ledger: How to Train on Low VRAM GPUs](#ledger-how-to-train-on-low-vram-gpus) &emsp; => A card with 24GB *(e.g. NVIDIA GeForce RTX 3090 and 4090)*
  - [:ledger: How to Train on Larger VRAM GPUs](#ledger-how-to-train-on-larger-vram-gpus) &emsp; => A Consumer GPU with 32GB *(e.g. NVIDIA GeForce RTX 5090)* &emsp; A Professional-Grade GPU with 40GB-48GB *(e.g. NVIDIA A100-40GB, A800-40GB, L20, and RTX A6000).*
  - [:ledger: How to Train on Sufficient VRAM GPUs](#ledger-how-to-train-on-sufficient-vram-gpus) &emsp; => Professional-Grade GPUs with ≥80GB *(e.g. NVIDIA A100-80GB, A800-80GB, H100, H800, H20-NVLink, and GB200).*
- [:mechanical_arm: Inference](#mechanical_arm-inference)
  - [:books: Related File for Inference](#books-related-file-for-inference)
  - [🤗 Checkpoint of VLA-Adapter](#ckpts)
  - [:notebook: How to Eval](#evals)
- [🌈 Success Rate Comparison](#results)
- [📝 Citation](#cite)
- [:heart: Acknowledgment](#heart-acknowledgment)

<br/>

## :rocket: Quick Start


### Conda Environment of VLA-Adapter

```bash
# Create and activate conda environment
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter
```

### Install Dependencies

```bash
# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone vla-adapter repo and pip install to download dependencies
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
cd VLA-Adapter
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install "flash-attn==2.5.5" --no-build-isolation
# If you run into difficulty, try `pip cache remove flash_attn` first, or visit the
# website to download it. (https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5)
# You can download the corresponding `.whl` file according to the cuda version of `nvidia-smi`,
# and then run `pip install flash_attn-2.5.5+cuXX...whl` to install it. 
# We use the `flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` file.
```

<br/>
<br/>


## :pencil: Data Preparation

### LIBERO Benchmark

- **(Optional)**

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From vla-adapter base dir
```

To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning experiments, run the command below. This will download the `Spatial`, `Object`, `Goal`, and `Long` datasets in `RLDS` format, i.e., `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`. (`"_no_noops"` stands for no no-op actions, i.e., training samples with near-zero actions are filtered out). These datasets require `~10GB` of memory in total. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup). You can use these to fine-tune Prismatic-VLMs (built on Qwen2.5-0.5B) or other VLMs.

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

🌟 Attention! The dataset downloaded in this way needs to remove of the ``modified_`` word to adapt to the path of - [:pushpin: Benchmark Location](#pushpin-benchmark-location)!!!

When using LIBERO, you may get an error message like `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`. You can use:

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

### CALVIN Benchmark

- **(Optional)**

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT

# Installation of `pyhash` may fail on some machines. If it fails, you can solve it by lowering the `setuptools` version: `pip install setuptools==57.5.0`
sh install.sh
```

To download the [CALVIN ABC→D datasets](https://github.com/mees/calvin/tree/main/dataset) that we used in our fine-tuning experiments, run the command below. 

```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```

If you want to download the RLDS format, you can visit [here](https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds) to download it. This dataset requires `~50GB` of memory.

When using CALVIN, you may get an error message like `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`. You can use:

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```


### :video_game: Our Dependencies 

- **(including LIBERO and CALVIN)**

At this point, the environment is fully installed. If you want to confirm whether the environment is correct, you can see the `our_envs.txt` file we released.


### :pushpin: Benchmark Location

The downloaded dataset can be placed in the `/data` folder. The overall directory structure is as follows:

```
·
├── data
·   ├── libero
    │   ├── libero_10_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_goal_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    │   ├── libero_object_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_spatial_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    │
    ├── calvin_abc
    │   └── 1.0.0  (It contains some json files, 512 train tfrecord files, and 32 valid tfrecord files)
    │
    └── other benchmarks ...
```

<br/>
<br/>

## ⚓ VLM backbone <a name="vlm"></a>
We use the `Prismatic-VLMs` architecture. Since the file is large, please download it from [here](https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b). Then put it in the `/pretrained_models` folder. The file structure is:

```
·
├── pretrained_models
·   ├── configs
    └── prism-qwen25-extra-dinosiglip-224px-0_5b
```


<br/>
<br/>

## :fire: Training for Different Configurations

**We provide different training configurations for different users. You can choose the configuration suitable for training based on your GPU card type.**

### :books: Related File for Training
* `vla-scripts/finetune.py`: VLA fine-tuning script


### :ledger: How to Train on Extremely Limited VRAM GPUs

***=> Extremely Limited VRAM (A card with 10GB-12GB) (e.g. NVIDIA GeForce RTX 2080Ti, 3060, 3080, 4070, 4080, and 5070).***

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If your resources are extremely limited, you can set `--batch_size 1` and `--lora_rank 64`, it only requires `9.6GB` of VRAM. Certainly, `batch size = 1` will cause gradient updates to be greatly affected by extreme values, and loss convergence will be unstable. In this case, you can modify the `grad_accumulation_steps` parameter to simulate a similar effect. For example, `--batch_size 1` with `--grad_accumulation_steps 8` has a similar effect to `--batch_size 8`, but the training speed will be slower. This means that you can't use the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) model on a card with `10GB` because even with `batch size = 1`, it requires `25GB` of VRAM. Fortunately, you can use VLA-Adapter. However, the `batch size` is still small, you can increase `--max_steps` to achieve the performance reported in the paper.

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version`, requiring only `8.6GB` of VRAM. You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

 ```bash
data_name=libero_spatial_no_noops

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
--num_steps_before_decay 400000 \
--max_steps 400005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 1 \
--grad_accumulation_steps 8 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.

<br/>

### :ledger: How to Train on Low VRAM GPUs

***=> Low VRAM (A card with 24GB) (e.g. NVIDIA GeForce RTX 3090 and 4090).***

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If you have such a device, you can increase the `batch size` and `lora rank`: `--batch_size 4` and `--lora_rank 64`. This only takes nearly `20GB`. This is consistent with the rank in our paper. This means that you can't use the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) model on a card with `24GB` because even with `batch size = 1`, it requires `25GB` of VRAM. Fortunately, you can use VLA-Adapter. However, the `batch size` is still small, you can increase `--max_steps` to achieve the performance reported in the paper.

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch), requiring only `17.6GB` of VRAM. You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.


 ```bash
data_name=libero_spatial_no_noops

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
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.



<br/>

### :ledger: How to Train on Larger VRAM GPUs

***=> A Consumer GPU with 32GB (e.g. NVIDIA GeForce RTX 5090) <br/> => A Professional-Grade GPU with 40GB-48GB (e.g. NVIDIA A100-40GB, A800-40GB, L20, and RTX A6000).***


>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If you have such a device, you can increase the `batch size` and `lora rank`: `--batch_size 8` and `--lora_rank 64`. This only takes nearly `29GB`. 

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

With this configuration, you can achieve the same results as in our paper on the `LIBERO-Object` benchmark, achieving a `99.2%` success rate, in just `8 hours`. The `LIBERO-Spatial` benchmark requires approximately 10 hours of training. However, the `LIBERO-Long` benchmark takes longer because its tasks are longer and more difficult, requiring more training steps to achieve superior performance.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch). You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

 ```bash
data_name=libero_spatial_no_noops

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
--batch_size 8 \
--grad_accumulation_steps 2 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.



<br/>

### :ledger: How to Train on Sufficient VRAM GPUs

***=> Professional-Grade GPUs with ≥80GB (e.g. NVIDIA A100-80GB, A800-80GB, H100, H800, H20-NVLink, and GB200).***

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

You can use 1 to 8 GPUs for training by changing the number of `CUDA_VISIBLE_DEVICES` to the GPU number and the number of GPUs after `--nproc-per-node`. In our paper, we use 4×H100 GPU for training. In this configuration, the four suites of the LIBERO benchmark, `Spatial` (only five hours), `Object` (less than one hour), `Goal` (three hours), and `Long` (half a day); the `CALVIN` benchmark (eight hours)

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.


>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch). You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
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
--num_steps_before_decay 150000 \
--max_steps 150005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 16 \
--grad_accumulation_steps 1 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--spatial--$current_time \
> logs/VLA-Adapter--spatial--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.

## :mechanical_arm: Inference

### :books: Related File for Inference
* `experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO eval script
  * `libero_utils.py`: LIBERO eval utils
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: VLA-specific eval utils
  * `robot_utils.py`: Other eval utils

<br/>

### 🤗 Checkpoint of VLA-Adapter <a name="ckpts"></a>
We fine-tuned `Qwen2.5-0.5B` with our adapter bridge paradigm on four LIBERO task suites independently: `LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, and `LIBERO-Long`. 
The four VLA-Adapter checkpoints for LIBERO are available on Hugging Face:
* [VLA-Adapter/LIBERO-Spatial](https://huggingface.co/VLA-Adapter/LIBERO-Spatial) 
* [VLA-Adapter/LIBERO-Object](https://huggingface.co/VLA-Adapter/LIBERO-Object)
* [VLA-Adapter/LIBERO-Goal](https://huggingface.co/VLA-Adapter/LIBERO-Goal)
* [VLA-Adapter/LIBERO-Long](https://huggingface.co/VLA-Adapter/LIBERO-Long)

In addition, we also provide a `Pro` version, we used `4*H100` GPUs for training, `--batch_size 16`, `--lora rank 64`, and the `--max_steps 100000`. The Pro checkpoints is:

* [VLA-Adapter/LIBERO-Spatial-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Spatial-Pro) `(97.8 -> 99.6)`
* [VLA-Adapter/LIBERO-Object-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Object-Pro) `(99.2 -> 99.6)`
* [VLA-Adapter/LIBERO-Goal-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Goal-Pro) `(97.2 -> 98.2)`
* [VLA-Adapter/LIBERO-Long-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Long-Pro) `(95.0 -> 96.4)`
* [VLA-Adapter/CALVIN-ABC-Pro](https://huggingface.co/VLA-Adapter/CALVIN-ABC-Pro) `(4.42 -> 4.50)`

These files need to be placed in the `/output` folder. If you trained your own models, it will also be stored here. The subsequent eval code will call the model in this folder for inference.


<br/>


### :notebook: How to Eval <a name="evals"></a>

**We strongly recommend that you use our open source `Pro` version of the model, which has stronger performance.** To start evaluations with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above. If you want to use the original version of the model, you only need to adjust the `-- use_pro_version` parameter to `False` and pass the original version of the model to the `--pretrained_checkpoint` parameter. Finally, the inference results will be displayed in the `/eval_logs` folder, and the inference video will be displayed in the `/rollouts/vla-adapter` folder. 


```bash
# Launch LIBERO-Spatial-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  > eval_logs/Spatial--chkpt.log 2>&1 &


# Launch LIBERO-Object-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object \
  --use_pro_version True \
  > eval_logs/Object--chkpt.log 2>&1 &


# Launch LIBERO-Goal-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Goal-Pro \
  --task_suite_name libero_goal \
  --use_pro_version True \
  > eval_logs/Goal--chkpt.log 2>&1 &


# Launch LIBERO-Long-Pro (LIBERO-10) evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-long-Pro \
  --task_suite_name libero_10 \
  --use_pro_version True \
  > eval_logs/Long--chkpt.log 2>&1 &


# Launch CALVIN ABC→D-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint outputs/CALVIN-ABC-Pro \
  > eval_logs/CALVIN--ABC.log 2>&1 &
```

If you want to get the inference **throughput**, you can run it in the `run_libero_eval.py` file. You can add  `start = time.time()` and `end = time.time()` before and after `lines 334--345` and calculate the difference between the two. This difference is the time it takes to generate `8 chunks`. This gives you the inference throughput. We measured it multiple times and took the average value of `0.036s`.

<br/>

## 🌈 Success Rate Comparison <a name="results"></a>

All our results are inferred on `H100`. You can find the inference `log` file in the model released on [HF](https://huggingface.co/VLA-Adapter) for viewing. The evaluation script will run 500 trials by default (10 tasks x 50 episodes each) in LIBERO and 1,000 task sequences in CALVIN. Use the same card for training and inference whenever possible. **Note that results may vary slightly if you use a different GPU than the H100.** This phenomenon is also mentioned in the OpenVLA-OFT readme file.

### Performance on LIBERO benchmark. 

<b><i>XX</i></b> represents the best performance, <b>XX</b> represents the second best performance, and <i><u>XX*</u></i> represents the third best performance.
<table>
  <tr>
   <td><strong>LIBERO</strong></td>  <td><strong>Methods</strong></td>
   <td><strong>Scale</strong></td>  <td><strong>Spatial</strong></td>
   <td><strong>Object</strong></td>  <td><strong>Goal</strong></td>
   <td><strong>Long</strong></td>  <td><strong>Avg.</strong></td>
  </tr>

  <tr><td rowspan="10">Large-scale</td><td>FlowVLA (Zhong et al., 2025)</td>
   <td>8.5B</td><td>93.2</td><td>95.0</td><td>91.6</td><td>72.6</td><td>88.1</td></tr>

  <tr><td>UnifiedVLA (Wang et al., 2025)</td>
   <td>8.5B</td><td>95.4</td><td><i><u>98.8*</u></i></td><td> 93.6 </td><td>94.0 </td><td>95.5</td></tr>

  <tr><td>OpenVLA (Kim et al., 2024)</td>
   <td>7B</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr>

  <tr><td>OpenVLA-OFT (Kim et al., 2025)</td>
   <td>7B</td><td><i><u>97.6*</u></i></td><td>98.4</td><td><b>97.9</b></td><td><i><u>94.5*</u></i></td><td><i><u>97.1*</u></i></td></tr>

  <tr><td>UniVLA (Bu et al., 2025)</td>
   <td>7B</td><td>96.5</td><td> 96.8</td><td> 95.6 </td><td>92.0 </td><td>95.2</td></tr>

  <tr><td>CoT-VLA (Zhao et al., 2025)</td>
   <td>7B</td><td>87.5 </td><td>91.6 </td><td>87.6</td><td> 69.0</td><td> 81.1</td></tr>

  <tr><td>WorldVLA (Cen et al., 2025)</td>
   <td>7B</td><td>87.6</td><td> 96.2</td><td> 83.4</td><td> 60.0</td><td> 81.8</td></tr>

  <tr><td>TraceVLA (Zheng et al., 2025)</td>
   <td>7B</td><td>84.6</td><td> 85.2</td><td> 75.1</td><td> 54.1</td><td> 74.8</td></tr>

  <tr><td>MolmoAct (Lee et al., 2025)</td>
   <td>7B</td><td>87.0</td><td> 95.4 </td><td>87.6</td><td> 77.2 </td><td>86.6</td></tr>

  <tr><td>ThinkAct (Huang et al., 2025)</td>
   <td>7B</td><td>88.3 </td><td>91.4</td><td> 87.1</td><td> 70.9</td><td> 84.4</td></tr>

  <tr><td rowspan="7">Small-scale</td><td>4D-VLA (Zhang et al., 2025)</td>
   <td>4B</td><td>88.9</td><td> 95.2</td><td> 90.9</td><td> 79.1 </td><td>88.6</td></tr>

  <tr><td>SpatialVLA (Qu et al., 2025)</td>
   <td>4B</td><td>88.2</td><td> 89.9</td><td> 78.6</td><td> 55.5 </td><td>78.1</td></tr>

  <tr><td>π0 (Black et al., 2024)</td>
   <td>3B</td><td>96.8</td><td><i><u>98.8*</u></i></td><td>95.8</td><td> 85.2</td><td> 94.2</td></tr>

  <tr><td>π0-FAST (Pertsch et al., 2025)</td>
   <td>3B</td><td>96.4</td><td> 96.8 </td><td>88.6</td><td> 60.2</td><td> 85.5</td></tr>

  <tr><td>NORA (Hung et al., 2025)</td>
   <td>3B</td><td>92.2 </td><td>95.4 </td><td>89.4</td><td> 74.6 </td><td>87.9</td></tr>

  <tr><td>SmolVLA (Shukor et al., 2025)</td>
   <td>2.2B</td><td>93.0</td><td> 94.0 </td><td>91.0</td><td> 77.0 </td><td>88.8</td></tr>

  <tr><td>GR00T N1 (NVIDIA et al., 2025)</td>
   <td>2B</td><td>94.4</td><td> 97.6 </td><td>93.0 </td><td>90.6</td><td> 93.9</td></tr>

  <tr><td rowspan="5">Tiny-scale</td><td>Seer (Tian et al., 2025)</td>
   <td>0.57B</td><td>-</td><td> - </td><td>- </td><td>78.7</td><td> 78.7</td></tr>

  <tr><td>VLA-OS (Gao et al., 2025)</td>
   <td>0.5B</td><td>87.0 </td><td>96.5</td><td> 92.7 </td><td>66.0</td><td> 85.6</td></tr>

  <tr><td>Diffusion Policy (Chi et al., 2023)</td>
   <td>-</td><td>78.3</td><td> 92.5</td><td> 68.3 </td><td>50.5 </td><td>72.4</td></tr>

  <tr><td><b>VLA-Adapter (Ours)</b></td>
   <td><b>0.5B</b></td><td><b>97.8</b></td><td><b>99.2</b></td><td><i><u>97.2*</u></i></td><td> <b>95.0 </b></td><td><b>97.3</b></td></tr>

  <tr><td><b>VLA-Adapter-Pro (Ours)</b></td>
   <td><b>0.5B</b></td><td><b><i>99.6</i></b></td><td><b><i>99.6</i></b> </td><td><b><i>98.2</i></b></td><td><b><i>96.4</i></b></td><td><b><i>98.5</i></b></td></tr>
  
</table>

### Performance on CALVIN ABC→D benchmark. 

<b><i>XX</i></b> represents the best performance, <b>XX</b> represents the second best performance, and <i><u>XX*</u></i> represents the third best performance.

<table>
  <tr>
   <td><strong>CALVIN</strong></td>  <td><strong>Methods</strong></td>
   <td><strong>Scale</strong></td>  <td><strong>1</strong></td>
   <td><strong>2</strong></td>  <td><strong>3</strong></td>
   <td><strong>4</strong></td>  <td><strong>5</strong></td> <td><strong>Avg. len</strong></td>
  </tr>

  <tr><td rowspan="8">Large-scale</td><td>UniVLA (Bu et al., 2025) </td><td>7B </td><td>95.5 </td><td>85.8 </td><td>75.4</td><td> 66.9 </td><td>56.5 </td><td>3.80</tr>

  <tr><td>OpenVLA (Kim et al., 2024) </td><td> 7B</td><td> 91.3</td><td> 77.8 </td><td>62.0 </td><td>52.1 </td><td>43.5</td><td> 3.27</td></tr>

  <tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td> 7B</td><td> 96.3</td><td> 89.1 </td><td>82.4</td><td> 75.8</td><td> 66.5</td><td> 4.10</td></tr>

  <tr><td>VLAS (Zhao et al., 2025b) </td><td> 7B</td><td> 87.2 </td><td>64.2</td><td> 40.9 </td><td>28.1</td><td> 19.6 </td><td>2.40</td></tr>

  <tr><td>LCB (Shentu et al., 2024) </td><td> 7B</td><td> 73.6 </td><td>50.2 </td><td>28.5 </td><td>16.0 </td><td>9.9 </td><td>1.78</td></tr>

  <tr><td>RoboDual (Bu et al., 2024a) </td><td> 7B</td><td> 94.4</td><td> 82.7</td><td> 72.1</td><td> 62.4 </td><td>54.4</td><td> 3.66</td></tr>

  <tr><td>OpenHelix (Cui et al., 2025)  </td><td> 7B</td><td> <i><u>97.1*</u></i> </td><td>91.4 </td><td>82.8</td><td> 72.6</td><td> 64.1 </td><td>4.08</td></tr>

  <tr><td>ReconVLA (Song et al., 2025c)  </td><td> 7B</td><td> 95.6 </td><td>87.6 </td><td>76.9</td><td> 69.3</td><td> 64.1 </td><td>3.95</td></tr>

  <tr><td rowspan="4">Small-scale</td><td>DeeR (Yue et al., 2024) </td><td> 3B</td><td> 86.2</td><td> 70.1 </td><td>51.8</td><td> 41.5</td><td> 30.4 </td><td>2.82</td></tr>

  <tr><td>RoboFlamingo (Li et al., 2024b) </td><td> 3B</td><td> 82.4 </td><td>61.9</td><td> 46.6 </td><td>33.1</td><td> 23.5</td><td> 2.48</td></tr>

  <tr><td>VPP (Hu et al., 2025)</td><td>  1.5B</td><td>  95.7</td><td>  91.2</td><td>  <i><u>86.3*</u></i></td><td>  <i><u>81.0*</u></i></td><td>  <i><u>75.0*</u></i></td><td>  <i><u>4.33*</u></i></td></tr>

  <tr><td>SuSIE (Black et al., 2024)</td><td>1.3B</td><td> 87.0</td><td> 69.0</td><td> 49.0 </td><td>38.0</td><td> 26.0</td><td> 2.69</td></tr>

  <tr><td rowspan="5">Tiny-scale</td><td>Seer-Large (Tian et al., 2025)</td><td>0.57B</td><td> 96.3 </td><td><i><u>91.6*</u></i></td><td> 86.1 </td><td>80.3 </td><td>74.0</td><td> 4.28</td></tr>

  <tr><td>MoDE (Reuss et al., 2025) </td><td> 0.44B </td><td>96.2</td><td> 88.9</td><td> 81.1</td><td> 71.8 </td><td>63.5 </td><td>4.01</td></tr>

  <tr><td>Seer (Tian et al., 2025) </td><td> 0.32B</td><td> 94.4 </td><td>87.2 </td><td>79.9 </td><td>72.2 </td><td>64.3</td><td> 3.98</td></tr>

  <tr><td><b>VLA-Adapter (Ours)</b></td>
   <td><b>0.5B</b></td><td><b><i>99.1</i></b> </td><td><b>94.6</b> </td><td><b>88.8</b></td><td> <b>82.8</b> </td><td><b>76.5</b> </td><td><b>4.42</b></td></tr>

  <tr><td><b>VLA-Adapter-Pro (Ours)</b></td>
   <td><b>0.5B</b></td><td><b>98.5</b></td><td><b><i>95.0</i></b> </td><td><b><i>90.5</i></b></td><td><b><i>85.3</i></b></td><td><b><i>80.0</i></b></td><td><b><i>4.50</i></b></td></tr>
  
</table>


<br/>


## 📝 Citation <a name="cite"></a>

### 🫶 If you feel that this paper, models, or codes are helpful, please cite our paper, thanks for your support of VLA-Adapter!

```bibtex
@article{wang2025vlaadapter,
  author={Wang, Yihao and Ding, Pengxiang and Li, Lingxiao and Cui, Can and Ge, Zirui and Tong, Xinyang and Song, Wenxuan and Zhao, Han and Zhao, Wei and Hou, Pengxu and Huang, Siteng and Tang, Yifan and Wang, Wenhui and Zhang, Ru and Liu, Jianyi and Wang, Donglin},
  title={VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  journal={arXiv preprint arXiv:2509.09372},
  year={2025}
}
```

## :heart: Acknowledgment

We thank [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), and [RoboDual](https://github.com/OpenDriveLab/RoboDual) for their open-sourced work!

## 🌟 Star History

<a href="https://www.star-history.com/#OpenHelix-Team/VLA-Adapter&Date">
  <img src="https://api.star-history.com/svg?repos=OpenHelix-Team/VLA-Adapter&type=Date" width="400" height="250" />
</a>

