import gc
import platform

import numpy as np
# import gradio as gr
import json 
import torch
import torchaudio
import librosa
import pandas as pd
# from msclap import CLAP

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
# from safetensors.torch import load_file
# from torch.nn import functional as F
from torchaudio import transforms as T
import os

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict

# model = None
# sample_rate = 16000
# sample_size = 160000


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        #model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")

    return model, model_config

def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=10,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1,
        save_name='output.wav'
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Return fake stereo audio
    conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size

    if negative_prompt:
        negative_conditioning = [{"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size
    else:
        negative_conditioning = None
        
    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        init_audio, in_sr = torchaudio.load(init_audio)
        # Turn into torch tensor, converting from int16 to float32
        # init_audio = torch.from_numpy(init_audio).float().div(32767)
        init_audio = init_audio.float().div(32767)
        # print(init_audio.shape)
        
        # if init_audio.dim() == 1:
        #     init_audio = init_audio.unsqueeze(0) # [1, n]
        # elif init_audio.dim() == 2:
        #     init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # If inpainting, send mask args
    # This will definitely change in the future
    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 

    # Do the audio generation
    audio = generate_diffusion_cond(
        model, 
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=input_sample_size,
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        mask_args = mask_args,
        callback = progress_callback if preview_every is not None else None,
        scale_phi = cfg_rescale
    )

    # Convert to WAV file
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # print(len(audio))
    # print(seconds_total)
    # print(sample_rate)
    # print(int(seconds_total*sample_rate))
    # print(len(audio))
    # print(audio.shape)
    audio = audio[:, :int(seconds_total*sample_rate)]
    torchaudio.save(save_name, audio, sample_rate)

    return save_name


def generate_aug_one_sample(model_config, duration, caption, steps=100, inpainting=False, init_audio=None, init_noise_level=80, output_file_name='output.wav'):

    prompt = caption
    negative_prompt = None

    model_conditioning_config = model_config["model"].get("conditioning", None)

    has_seconds_start = False
    has_seconds_total = False

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True

    if has_seconds_total:
        seconds_start_slider = 0
        seconds_total_slider = duration

    steps_slider = steps
    preview_every_slider = 0
    cfg_scale_slider = 10
    seed_textbox = -1
            
    sampler_type_dropdown = "dpmpp-3m-sde" #["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"]
    sigma_min_slider =  0.03 #gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
    sigma_max_slider = 500 #gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")
    cfg_rescale_slider = 0.0 #gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")

    if inpainting: 
        # Inpainting Tab
        sigma_max_slider.maximum=1000
        
        init_audio_checkbox = True
        init_audio_input = init_audio #gr.Audio(label="Init audio")
        init_noise_level_slider = init_noise_level #gr.Slider(minimum=0.1, maximum=100.0, step=0.1, value=80, label="Init audio noise level", visible=False) # hide this

        mask_cropfrom_slider = 0 #gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Crop From %")
        mask_pastefrom_slider = 0 #gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Paste From %")
        mask_pasteto_slider = 100 #gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Paste To %")

        mask_maskstart_slider = 50 #gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=50, label="Mask Start %")
        mask_maskend_slider = 100 #r.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Mask End %")
        mask_softnessL_slider = 0 #gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Left Crossfade Length %")
        mask_softnessR_slider = 0 #gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Right Crossfade Length %")
        mask_marination_slider = 0 #gr.Slider(minimum=0.0, maximum=1, step=0.0001, value=0, label="Marination level", visible=False) # still working on the usefulness of this 

        _ = generate_cond(
            prompt,
            negative_prompt=None,
            seconds_start=seconds_start_slider,
            seconds_total=seconds_total_slider,
            cfg_scale=cfg_scale_slider,
            steps=steps_slider,
            preview_every=preview_every_slider,
            seed=seed_textbox,
            sampler_type=sampler_type_dropdown,
            sigma_min=sigma_min_slider,
            sigma_max=sigma_max_slider,
            cfg_rescale=cfg_rescale_slider,
            use_init=init_audio_checkbox,
            init_audio=init_audio_input,
            init_noise_level=init_noise_level_slider,
            mask_cropfrom=mask_cropfrom_slider,
            mask_pastefrom=mask_pastefrom_slider,
            mask_pasteto=mask_pasteto_slider,
            mask_maskstart=mask_maskstart_slider,
            mask_maskend=mask_maskend_slider,
            mask_softnessL=mask_softnessL_slider,
            mask_softnessR=mask_softnessR_slider,
            mask_marination=mask_marination_slider,
            batch_size=1,
            save_name=output_file_name
            )
        
    else:
        # Default generation tab
        if init_audio is not None:
            init_audio_checkbox = True
        else:
            init_audio_checkbox = False
        init_audio_input = init_audio #r.Audio(label="Init audio")
        init_noise_level_slider = init_noise_level #gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

        _ = generate_cond(
            prompt,
            negative_prompt=None,
            seconds_start=seconds_start_slider,
            seconds_total=seconds_total_slider,
            cfg_scale=cfg_scale_slider,
            steps=steps_slider,
            preview_every=preview_every_slider,
            seed=seed_textbox,
            sampler_type=sampler_type_dropdown,
            sigma_min=sigma_min_slider,
            sigma_max=sigma_max_slider,
            cfg_rescale=cfg_rescale_slider,
            use_init=init_audio_checkbox,
            init_audio=init_audio_input,
            init_noise_level=init_noise_level_slider,
            mask_cropfrom=None,
            mask_pastefrom=None,
            mask_pasteto=None,
            mask_maskstart=None,
            mask_maskend=None,
            mask_softnessL=None,
            mask_softnessR=None,
            mask_marination=None,
            batch_size=1,
            save_name=output_file_name
            )

    return None


def create_augs(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False, json_path=None, output_folder=None, num_iters=5, use_label = "True", dataset_name = None, output_csv_path = './', num_process=0, init_noise_level=80, clap_filter="False", clap_threshold=75.0, initialize_audio = "True", dpo = "False"):

    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    # if clap_filter == "True":
    #     clap_model = CLAP(version = '2023', use_cuda=True)

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, device=device)
    
    model_type = model_config["model_type"]

    all_audios = []

    with open('/fs/nexus-projects/brain_project/aaai_2025/icassp_2025/single_audio_test.txt') as f:
        all_audios = f.readlines()

    all_audios = list(set([i.strip('\n') for i in all_audios]))

    old_audios_list = []
    new_audios_list = []
    new_labels_list = []
    new_caption_list = []

    import random
    # random_number = round(random.uniform(2, 4), 1)
    random_number = 10.0

    for aud in all_audios:

        output_file_name = '/fs/nexus-projects/brain_project/aaai_2025/icassp_2025/iclr/sonal_sounds_2/' + "_".join(aud.split(" ")).rstrip('_') + '.wav'

        generate_aug_one_sample(model_config, random_number, aud, steps=250, inpainting=False, init_audio=None, init_noise_level=init_noise_level, output_file_name=output_file_name)

        
    return None
