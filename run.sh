# data_name=calvin_abc_rlds
# beat_block_hammer_rt
# libero_object_no_noops
data_name=beat_block_hammer_rt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
CUDA_VISIBLE_DEVICES=3,4 torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir /home/ruihengwang/tensorflow_datasets \
    --dataset_name $data_name \
    --run_root_dir outputs \
    --use_film False \
    --num_images_in_input 3 \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --num_steps_before_decay 200000 \
    --max_steps 200005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_entity "my-wandb-org" \
    --wandb_project "$data_name" \
    --run_id_note VLA-Adapter--$data_name--$(date "+%Y_%m_%d_%H_%M_%S") \