# from ./tarin_diffusion_simpo.sh
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

accelerate launch ./train_diffusion_simpo.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --mixed_precision="fp16" \
  --train_batch_size=4 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=30000 \
  --checkpointing_steps=1000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=1600 \
  --learning_rate=1e-5 \
  --beta=200.0 \
  --gamma_beta_ratio=0.5 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --resolution=512 \
  --seed="0" \
  --run_validation --validation_steps=100 \
  --report_to="wandb" \
  --output_dir="SIMPO" \
  --validation_image_dir="SIMPO-val" \
  --push_to_hub
