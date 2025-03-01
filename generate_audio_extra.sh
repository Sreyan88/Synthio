#!/bin/bash
# set -x

input_jsonl=$1

python generate_audio_extra.py \
--ckpt-path /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/sonal_ft_2.safetensors \
--model-config /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/mod_config_false.json \
--input_json "$input_jsonl" \
--num_iters 1 \
--output_folder 'dummy' \
--use_label 'True' \
--dataset_name 'dummy' \
--output_csv_path 'dummy' \
--num_process 1 \
--init_noise_level 80.0 \
--initialize_audio 'True' \
--dpo 'True'