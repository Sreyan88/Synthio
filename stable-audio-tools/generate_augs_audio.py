from stable_audio_tools import get_pretrained_model
from stable_audio_tools.interface.generate_augs import create_augs
import json
import os

import torch

def main(args):
    torch.manual_seed(42)

    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    _ = create_augs(
        model_config_path = args.model_config, 
        ckpt_path=args.ckpt_path, 
        pretrained_name=args.pretrained_name, 
        pretransform_ckpt_path=args.ckpt_path, #args.pretransform_ckpt_path,
        model_half=args.model_half,
        json_path=args.input_json,
        output_folder=args.output_folder,
        num_iters=args.num_iters,
        use_label = args.use_label, 
        dataset_name = args.dataset_name, 
        output_csv_path = args.output_csv_path,
        num_process=args.num_process,
        init_noise_level=args.init_noise_level,
        clap_filter=None, 
        clap_threshold=None,
        initialize_audio=args.initialize_audio,
        dpo=args.dpo,
        supcon=args.supcon
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--share', action='store_true', help='Create a publicly shareable link', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    parser.add_argument('--num_iters', type=int, help='Number of augmentation iterations', required=True)
    parser.add_argument('--input_json', type=str, help='Path to input gold audios and captions', required=True)
    parser.add_argument('--output_folder', type=str, help='Path to save augmentations', required=True)
    parser.add_argument('--use_label', type=str, help='Whether to use labels or captions', required=True)
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--output_csv_path', type=str, help='Path to save csvs', required=True)
    parser.add_argument('--num_process', type=int, help='Path to save csvs', required=True)
    parser.add_argument('--init_noise_level', type=float, help='Noise for audio initialization', required=True)
    parser.add_argument('--initialize_audio', type=str, help='Whether to initialize audio for generation or not', required=True)
    parser.add_argument('--dpo', type=str, help='Whether to generate a CSV for DPO or not', required=True)
    parser.add_argument('--supcon', type=str, help='Whether to generate a CSV for Supervised Contrastive or not', required=True)
    args = parser.parse_args()
    main(args)