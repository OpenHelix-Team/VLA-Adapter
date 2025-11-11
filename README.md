
### The  real-world ALOHA system implementation of **VLA-Adapter**.

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


## :pencil: Data Preparation

### ALOHA Benchmark
If needed, see details on how to record the real HDF5 datasets [here](https://github.com/tonyzhaozh/aloha).

After data collection is complete, you need to convert the HDF5 format data to RLDS format. Specific methods can be found [here](https://github.com/HenryLiukkk/rlds_dataset_builder).



### :pushpin: Benchmark Location

The downloaded dataset can be placed in the `/data` folder. The overall directory structure is as follows:

```
·
├── data
·   ├── aloha
    │   └── aloha_put_x_into_the_box_80_demos
    │       └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │
    └── other benchmarks ...
```

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
data_name=aloha_put_x_into_the_box_80_demos

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/aloha \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 50000 \
--max_steps 100005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 4 \
--grad_accumulation_steps 4 \
--learning_rate 1e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--aloha_put_x_into_the_box_80_demos--$current_time \
> logs/VLA-Adapter--aloha_put_x_into_the_box_80_demos--$current_time.log 2>&1 &

```
Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.

For other GPU configuration solutions, please refer to [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter).

<br/>

## :mechanical_arm: Inference


### :notebook: How to Eval <a name="evals"></a>

```bash
# First launch your own robotic arm, then run the following code
python experiments/robot/aloha/run_aloha_eval.py \
  --pretrained_checkpoint <your_chkpt_path> \
  --use_l1_regression True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key aloha_put_x_into_the_box_80_demos \
  --num_rollouts_planned 1 \
  --max_steps 500
```
<br/>


## :heart: Acknowledgment

We thank [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), [RoboDual](https://github.com/OpenDriveLab/RoboDual) and [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) for their open-sourced work!


