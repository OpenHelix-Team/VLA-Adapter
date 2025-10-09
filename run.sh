# data_name=calvin_abc_rlds
data_name=libero_10_no_noops
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
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
    --save_freq 20000 \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --use_pro_version True \
    --wandb_entity "my-wandb-org" \
    --wandb_project "$data_name" \
    --run_id_note VLA-Adapter--$data_name--$(date +%s) \
    # --resume True \
    # --resum_vla_path outputs/configs+libero_10_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_10_no_noops----100000_chkpt \
    # --resume_step 100000 \
# > logs/VLA-Adapter--$data_name--$current_time.log 2>&1 &