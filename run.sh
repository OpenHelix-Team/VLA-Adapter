# data_name=calvin_abc_rlds
data_name=libero_10_no_noops
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
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
    --use_3d True \
    --inject_layers all \
    --run_id_note VLA-Adapter--$data_name--$(date "+%Y_%m_%d_%H_%M_%S") \
    # --resume True \
    # --resum_vla_path outputs/configs+calvin_abc_rlds+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--calvin_abc_rlds--2025_10_12_19_33_45--110000_chkpt \
    # --resume_step 110000 \
    # > experiments/logs/Train--$data_name--$(date "+%Y_%m_%d_%H_%M_%S").log 2>&1 &