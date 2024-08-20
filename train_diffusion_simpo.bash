export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

accelerate launch ./train_diffusion_simpo.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --mixed_precision="fp16" \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=30000 \
  --checkpointing_steps=1000 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --learning_rate=1e-5 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --resolution=512 \
  --seed="0" \
  --run_validation --validation_steps=50 \
  --report_to="wandb" \
  --output_dir="SIMPO" \
  --push_to_hub