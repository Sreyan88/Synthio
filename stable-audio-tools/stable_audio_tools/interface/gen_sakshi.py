import torch
import pandas as pd
from stable_audio_tools import get_pretrained_model
import json

from stable_audio_tools.interface.generate_augs import generate_aug_one_sample, load_model
#  import generate_aug_one_sample, load_model

model_config_path = '/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/mod_config.json'
ckpt_path = '/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/sonal_ft.safetensors'
pretrained_name = None
pretransform_ckpt_path = None
model_half=False

device = torch.device("cuda")

df = pd.read_csv('/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/dummy.csv')
if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)


for i, row in df.iterrows():

    caption = row['caption']
    duration = float(row['duration'])
    output_path = row['output_path']

    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, device=device)

    generate_aug_one_sample(model_config, duration, caption, steps=250, inpainting=False, init_audio=None, init_noise_level=80.0, output_file_name=output_path)


# caption,duration,output_path
# 'sound of a beep',2.0,'/fs/nexus-projects/brain_project/aaai_2025/stable-audio-tools/stable_audio_tools/interface/output_sakshi/beep.wav'