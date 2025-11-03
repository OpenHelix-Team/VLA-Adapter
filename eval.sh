export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/configs+libero_10_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_10_no_noops--2025_10_27_17_22_41--use_3d_True_dim_2048_inject_all--170000_chkpt \
  --task_suite_name libero_10 \
  --use_pro_version True \
  --use_3d True \
  --inject_layers all \
# outputs/configs+libero_10_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_10_no_noops--1759126170--160000_chkpt \  
  # > eval_logs/Spatial--chkpt.log 2>&1 &