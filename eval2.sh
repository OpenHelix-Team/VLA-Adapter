export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
CUDA_VISIBLE_DEVICES=7 python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint outputs/CALVIN-ABC-Pro
  # --pretrained_checkpoint outputs/configs+calvin_abc_rlds+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--calvin_abc_rlds----100000_chkpt \