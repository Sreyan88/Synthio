# Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data

This repository contains code for the ICLR 2025 Paper: [Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data](https://openreview.net/forum?id=bR1J7SpzrD)

As part of our ICLR 2025 paper, we are open-sroucing a modular pipeline for generating, augmenting, and training (small-scale) audio classification datasets using synthetic data. It is designed for low-resource scenarios where data scarcity is a bottleneck for model performance. The system supports iterative augmentation, supervised contrastive learning, and DPO-based fine-tuning, and integrates with models like AST and CLAP.

---

## 🧠 Overview

This repository provides tools to:
- Stratify and prepare low-resource datasets
- Generate captions using GAMA or other captioning tools
- Fine-tuning pre-trained Text-to-Audio Diffusion Models ([Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) + DPO)
- Filter audio using CLAP-based semantic similarity
- Train audio classifiers (e.g., AST)

---

## 🚀 Quick Start

### Quick Links:
- Our pre-trained Text-to-Audio Model (based on Stable Audio) - [Ckpt](https://huggingface.co/sonalkum/synthio-stable-audio-open) / [Space](https://huggingface.co/spaces/sonalkum/synthio-stable-audio-open)
- The CLAP model used in our experiments - [T5 Model](https://huggingface.co/sonalkum/synthio-t5) / Full Model

### Clone the repository:
```bash
git clone https://github.com/Sreyan88/Synthio.git
cd Synthio
```

### Install dependencies:
This project uses multiple conda environments. Please set them up first (preferabbly with the same name). We provide our own conda envrionments. However, one may also use the requirements.txt files:
- `stable_audio` – the main pipeline - [stable-audio env](extras/stable_audio_env)
- `gama` – for generating audio captions - please clone the [original repo](https://github.com/Sreyan88/GAMA) and install [requirements](https://github.com/Sreyan88/GAMA/blob/main/requirements.txt) starting from a new env (call it `gama`) - next, download the ckpt you would like to use and change line 231 in [this python file](./gama_csv_inf.py)
- `clap` / `msclap` – used for audio filtering - [clap env](extras/clap_env) , [msclap env](extras/msclap_env)
- `ast` – used for classifier training - [ast env](extras/ast_env)

If you happened to change any of the env names, please change them in `run.sh` too.

### Prepare your CSV files:
Place your dataset splits inside the `dataset_csvs/` folder:
- `tut_train.csv`
- `val.csv`
- `test.csv`

### Run the pipeline:
The entire pipeline, from start to end (please refer to the paper for more details), can be run using a single command:
```bash
sh run.sh
```

---

## 📄 Script Overview: `run.sh`

This main orchestration script handles the entire pipeline, auto-detects available GPUs, and distributes jobs.

---

## 🔧 Key Variables in `run.sh`

| Variable                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `dataset_name`            | Dataset name for naming output folders                                       |
| `input_train_file`        | Path to the full training CSV                                                |
| `valid_csv`, `test_csv`   | Paths to validation and test CSVs                                           |
| `num_samples`             | Number of samples for low-resource simulation (e.g., 100)                              |
| `num_iters`               | Augmentations per sample (e.g., 2)                                                   |
| `output_folder`           | Synthetic audio storage directory                                           |
| `output_csv_path`         | CSV metadata storage                                                        |
| `supcon`                  | Enables supervised contrastive learning                                     |
| `augment`                 | If True, enables augmentation pipeline, else directly trains a baseline w/o augmentation                                               |
| `dpo`                     | Enables DPO fine-tuning                                                     |
| `use_label`               | If True, generates captions with labels and **not** captions (e.g., "Sound of a dog")                    |
| `plain_caption`           | If True, uses simple GPT captions                                                         |
| `plain_wo_caption`        | GPT captions without labels                                                 |
| `use_ast`                 | Enables AST classifier training                                             |
| `clap_filter`             | Enables CLAP-based filtering                                                |
| `initialize_audio`        | If True, uses noise + audio in the diffusion forward pass                                      |
| `force_steps`             | Forces new file regeneration (when re-running an experiment)                                                   |
| `only_synthetic`          | If True, only synthetic data will be used for training (no gold data will be added)                                      |
| `dpo_ckpt_folder`         | Path forsaving DPO checkpoints                                                    |

---

## 📂 Repository Structure

```
Synthio/
├── ast/                        # AST classifier training code
├── dataset_csvs/               # Dataset CSV files
├── stable-audio-tools/         # Synthetic audio generation (diffusion model)
├── GAMA/                       # GAMA caption generation **(need to clone seperately)**
├── run.sh                      # Main orchestration script
├── generate_captions_gpt.py    # GPT captioning script
├── stratify_dataset.py         # Dataset stratification
├── merge_csv.py, split_csvs.py, filter_audios.py, etc.
```

---

## 🧪 Expected Output

After running `sh run.sh`, you will get:
- Synthetic audio files in `./tut_urban_synthetic/`
- Merged metadata CSVs in `./tut_urban/`
- AST trained models/checkpoints
- The terminal will print the scores on the test set

---

## ❓ FAQ

**Q:** Train only on real data?
- Set `augment=False` and `only_synthetic=False`.

**Q:** Enable supervised contrastive learning?
- Set `supcon=True`. This is an additional feature.

**Q:** Any additional problem?
- Please raise an issue.

---

## 📜 License

MIT License

---

## ✨ Acknowledgments

- [GAMA](https://github.com/Sreyan88/GAMA)
- [CLAP](https://github.com/LAION-AI/CLAP)
- [AST](https://github.com/YuanGongND/ast)
- [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)
- [DPO](https://github.com/declare-lab/tango)

---

## 📖  Citations

If you use this work or any of its components, please cite:
```
@inproceedings{ghosh2025synthio,
  title={Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data},
  author={Sreyan Ghosh and Sonal Kumar and Zhifeng Kong and Rafael Valle and Bryan Catanzaro and Dinesh Manocha},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=bR1J7SpzrD}
}
```
