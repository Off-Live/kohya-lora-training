#!/bin/bash

accelerate launch train_network.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"\
   --dataset_config="./config.toml" \
   --output_dir="./model" \
   --logging_dir="./logs" \
   --log_with="tensorboard"\
   --output_name="your_lora" \
   --save_model_as=safetensors \
   --network_module=networks.lora \
   --seed 0 \
    --text_encoder_lr=0.000025 \
    --unet_lr=0.00005 \
    --network_dim=8  \
    --network_alpha=1 \
    --learning_rate=0.00005 \
    --lr_scheduler="cosine" \
    --max_train_steps="3000" \
    --save_every_n_epochs="1" \
    --mixed_precision="fp16" \
    --save_precision="fp16" \
    --optimizer_type="AdamW8bit" \
    --save_every_n_steps="75" \
    --sample_sampler=euler_a \
    --sample_prompts="./prompt.txt" \
    --sample_every_n_steps="75" 