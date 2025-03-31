#!/bin/bash

input_csv=$1
dataset=$2
num_samples=$3

input_json="/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/stable_audio_tools/configs/dataset_configs/local_training_example.json"

# input_config="/fs/nexus-projects/brain_project/microsoft/cache/hub/models--stabilityai--stable-audio-open-1.0/snapshots/4ab2b18994346363f65d4acbe7b034e814d99040/model_config.json"

input_config="/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/mod_config.json"
#"/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/config_adapter.json"

python convert_json.py --input_json $input_json --output_json "${input_json%.json}_${dataset}_${num_samples}.json" --new_path $1

python ./train.py --dataset-config "${input_json%.json}_${dataset}_${num_samples}.json" --model-config $input_config --pretrained-ckpt-path /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/sonal_ft_2.safetensors --save-dir /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/${dataset}_${num_samples} --name harmonai_train

#python ./train.py --dataset-config "${input_json%.json}_${dataset}_${num_samples}.json" --model-config $input_config --pretrained-ckpt-path /fs/nexus-projects/brain_project/microsoft/cache/hub/models--stabilityai--stable-audio-open-1.0/snapshots/4ab2b18994346363f65d4acbe7b034e814d99040/model.safetensors --save-dir /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/${dataset}_${num_samples} --name harmonai_train

last_checkpoint=$(ls -t "/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/${dataset}_${num_samples}" | head -n 1)

last_checkpoint_path=$(realpath "/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/${dataset}_${num_samples}/$last_checkpoint")

python unwrap_model.py --model-config $input_config --ckpt-path $last_checkpoint_path --name ${dataset}_${num_samples} --use-safetensors

rm /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/${dataset}_${num_samples}/*

# python unwrap_model.py --model-config /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/mod_config.json --ckpt-path /fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/epoch=36-step=660000.ckpt --name sonal_ft_3 --use-safetensors

# /fs/gamma-projects/audio/audio_datasets/epoch=14-step=270000.ckpt --name sonal_ft_3 --use-safetensors
# /fs/nexus-projects/brain_project/epoch=31-step=570000.ckpt