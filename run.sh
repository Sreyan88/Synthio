#!/bin/bash
# set -x

export HF_TOKEN='None'
export HF_HOME=./cache

dataset_name="tut_urban"
input_train_file="./dataset_csvs/tut_train.csv"
valid_csv="./dataset_csvs/val.csv"
test_csv="./dataset_csvs/test.csv"
domain="environmental_sounds"


output_folder="./${dataset_name}_synthetic/"
output_folder_supcon="./${dataset_name}_synthetic_supcon/"
num_iters=2 # number of augs for each sample to be generated
num_samples=200 #number of samples in the low-resource split
init_noise_level=80.0
initialize_audio=False
output_csv_path="./${dataset_name}/"
clap_threshold="0.85"
supcon=False
multi_label=False
# captioning arguments
use_label=True
plain_caption=False
plain_wo_caption=False
# iterative arguments
iterative=False
# encoder params
use_ast=True
clap_full_ft=False
# filter params
clap_filter=True
filter_w_finetune=False
full_finetune_clap="False"
clap_exp_name="aug_clap"
#run arguments
augment=True
force_steps=True
only_synthetic=True
# dpo parameters
dpo=True
use_dpo=True
dpo_ckpt_folder="./stable-audio-tools/" #path to save DPO checkpoint

# Check if the directory exists where we need to save synthetic audios
if [ -d "$output_folder" ]; then
  echo "Directory $output_folder already exists."
else
  echo "Directory $output_folder does not exist. Creating now..."
  mkdir -p "$output_folder"
  if [ $? -eq 0 ]; then
    echo "Directory $output_folder created successfully."
  else
    echo "Failed to create directory $output_folder."
  fi
fi

# Check if the directory exists where we need to save csv
if [ -d "$output_csv_path" ]; then
  echo "Directory $output_csv_path already exists."
else
  echo "Directory $output_csv_path does not exist. Creating now..."
  mkdir -p "$output_csv_path"
  if [ $? -eq 0 ]; then
    echo "Directory $output_csv_path created successfully."
  else
    echo "Failed to create directory $output_csv_path."
  fi
fi

# Check if the directory exists where we need to save supervised contrastive audios
if [ -d "$output_folder_supcon" ]; then
  echo "Directory $output_folder_supcon already exists."
else
  echo "Directory $output_folder_supcon does not exist. Creating now..."
  mkdir -p "$output_folder_supcon"
  if [ $? -eq 0 ]; then
    echo "Directory $output_folder_supcon created successfully."
  else
    echo "Failed to create directory $output_folder_supcon."
  fi
fi

# count GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# input your conda path
source /your/conda/path/miniconda3/bin/activate

# stratify the dataset
eval "$(conda shell.bash hook)"
conda activate stable_audio

if [ ! -f "${input_train_file%.csv}_$num_samples.csv" ]; then
    python stratify_dataset.py --input_csv $input_train_file --num_samples $num_samples --output_csv "${input_train_file%.csv}_$num_samples.csv" --dataset_name $dataset_name --multi_label "$multi_label"
fi

# store before changing
orig_input_train_file=$input_train_file
# assign input_train to new file
input_train_file="${input_train_file%.csv}_$num_samples.csv"

if [ "$augment" = True ]; then

    python split_csvs.py --input_csv $input_train_file --num $gpu_count

    # store file names in an array
    files=()
    for i in $(seq 0 $(($gpu_count - 1))); do
        files+=("${input_train_file%.csv}_$i.csv")
    done

    # store GPUs in a list
    gpus=()
    for i in $(seq 0 $(($gpu_count - 1))); do
        gpus+=($i)
    done

    # Generate captions for every instance
    if [ "$use_label" = False ] && [ "$plain_wo_caption" = False ]; then
        eval "$(conda shell.bash hook)"
        conda activate gama
        cd ./GAMA/
        cp gama_csv_inf.py GAMA/
        for i in $(seq 0 $(($gpu_count-1))); do
            CUDA_VISIBLE_DEVICES=${gpus[$i]} python gama_csv_inf.py --input_csv ${files[$i]} --output_csv ${files[$i]} &
        done
        cd ../
        eval "$(conda shell.bash hook)"
        conda activate stable_audio
    fi

    wait

    # finetune the model using DPO
    if [ "$dpo" = True ]; then
        if [ ! -f "$output_csv_path/${dataset_name}_dpo_merged.csv" ] || [ "$force_steps" = True ]; then
            cd stable-audio-tools
            for i in $(seq 0 $(($gpu_count-1))); do
                # hard coded to always use label (Sound of a X) for DPO training -- 4th arg is "True"
                CUDA_VISIBLE_DEVICES=${gpus[$i]} sh generate_augs_audio.sh ${files[$i]} 2 $output_folder "True" $dataset_name $output_csv_path $i $init_noise_level "False" "$dpo" "False" "None" "False" &
            done
            wait
            cd ../
        fi
        dpo=False #set dpo back to false for augmentation generation
        python merge_csv.py --output_csv_path $output_csv_path --dataset_name $dataset_name --num $gpu_count --clap_filter "False" --dpo "True"
        if [ ! -f "${dpo_ckpt_folder}${dataset_name}_${num_samples}.safetensors" ] || [ "$force_steps" = True ]; then
            cd stable-audio-tools
            sh finetune.sh "$output_csv_path/${dataset_name}_dpo_merged.csv" $dataset_name $num_samples
            wait
            cd ../
        fi
    fi

    wait

    if [ "$supcon" = True ]; then
        # first generate new captions using GPT for supervised contrastive
        for i in $(seq 0 $(($gpu_count-1))); do
            echo ${files[$i]}
            python generate_captions_gpt.py --input_csv ${files[$i]} --plain_caption "$plain_caption" --domain $domain --supcon "True" &
        done
        supcon_csv="$output_csv_path/${dataset_name}_supcon_merged.csv"
        if [ ! -f $supcon_csv ]; then
            # next generate audios for the new captions
            cd stable-audio-tools
            for i in $(seq 0 $(($gpu_count-1))); do
                # hard code $use_dpo to True and $dpo to False
                CUDA_VISIBLE_DEVICES=${gpus[$i]} sh generate_augs_audio.sh ${files[$i]} 4 $output_folder_supcon "False" $dataset_name $output_csv_path $i $init_noise_level "$initialize_audio" "False" "True" ${dpo_ckpt_folder}${dataset_name}_${num_samples}.safetensors "True" &
            done
            wait
            cd ../
        fi
        python merge_csv.py --output_csv_path $output_csv_path --dataset_name $dataset_name --num $gpu_count --clap_filter "False" --dpo "False" --supcon "True"
    fi

    # generate new captions using GPT
    if [ "$use_label" = False ]; then
        for i in $(seq 0 $(($gpu_count-1))); do
            echo "Generating GPT Captions"
            echo ${files[$i]}
            python generate_captions_gpt.py --input_csv ${files[$i]} --plain_caption "$plain_caption" --domain $domain --supcon "False" --plain_wo_caption "$plain_wo_caption" --multi_label "$multi_label" &
        done
    fi
    
    wait

    if [ ! -f "$output_csv_path/${dataset_name}_merged.csv" ] || [ "$force_steps" = True ]; then
    # generate final augmentations
        cd stable-audio-tools
        for i in $(seq 0 $(($gpu_count-1))); do
            CUDA_VISIBLE_DEVICES=${gpus[$i]} sh generate_augs_audio.sh ${files[$i]} $num_iters $output_folder "$use_label" $dataset_name $output_csv_path $i $init_noise_level "$initialize_audio" "False" "$use_dpo" ${dpo_ckpt_folder}${dataset_name}_${num_samples}.safetensors "False" &
        done
        wait
        cd ../
    fi

    # do clap filter
    if [ "$clap_filter" = True ]; then
        if [ "$filter_w_finetune" = True ]; then
            eval "$(conda shell.bash hook)"
            # input your anaconda path
            source /your/conda/path/anaconda3/bin/activate
            conda activate clap
            cd ./CLAP/src/laion_clap/
            echo "Training Dataset: $input_train_file"
            echo "Validation Dataset: $valid_csv"
            sh htsat-roberta-large-dataset-fusion.sh $clap_exp_name $input_train_file $valid_csv $full_finetune_clap
            wait
            for i in $(seq 0 $(($gpu_count-1))); do
                echo "Filtering"
                CUDA_VISIBLE_DEVICES=${gpus[$i]} python filter_audios_sonal.py --model_path "/fs/gamma-projects/audio/clap_logs/${clap_exp_name}/checkpoints/epoch_latest.pt" --input_csv_path $output_csv_path --clap_threshold "$clap_threshold" --dataset_name $dataset_name --iter $i --use_label "$use_label" &
            done
            cd /fs/nexus-projects/brain_project/aaai_2025/
            eval "$(conda shell.bash hook)"
            source /fs/nexus-projects/brain_project/miniconda3/bin/activate
            conda activate stable_audio
            wait
        else
            eval "$(conda shell.bash hook)"
            conda activate msclap
            for i in $(seq 0 $(($gpu_count-1))); do
                echo "Filtering"
                CUDA_VISIBLE_DEVICES=${gpus[$i]} python filter_audios.py --input_csv_path $output_csv_path --clap_threshold "$clap_threshold" --dataset_name $dataset_name --iter $i --use_label "$use_label" &
            done
            eval "$(conda shell.bash hook)"
            conda activate stable_audio
            wait
        fi
    fi

    # merge the training CSVs
    python merge_csv.py --output_csv_path $output_csv_path --dataset_name $dataset_name --num $gpu_count --clap_filter "$clap_filter"

    # iterative refinement (optional)
    if [ "$iterative" = True ]; then
        echo "Entering iterative refinement"
        while [ $counter -le 2 ]; do
          python merge_csv.py --output_csv_path $output_csv_path --dataset_name $dataset_name --num $gpu_count --clap_filter "False" --filteredout "True"
          wait
          exit 0
          python generate_captions_gpt.py --input_csv "$output_csv_path/${dataset_name}_filteredout_merged.csv" --plain_caption "False" --domain $domain --supcon "False" --plain_wo_caption "False" --multi_label "False" --iteration_stage "True"
          wait
          cd stable-audio-tools
          CUDA_VISIBLE_DEVICES=0 sh generate_augs_audio_extra.sh "$output_csv_path/${dataset_name}.iter.jsonl"
          wait
          cd ../
          CUDA_VISIBLE_DEVICES=0 python filter_audios.py --input_csv_path $output_csv_path --clap_threshold "$clap_threshold" --dataset_name $dataset_name --iter $i --use_label "$use_label"
          wait
          counter=$((counter+1))
        done
    fi

    # generate the label map for AST training
    python generate_label_map.py --dataset_name $dataset_name --input_csv "$output_csv_path/${dataset_name}_merged.csv" --output_json "/fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/${dataset_name}.json" --output_csv_path /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/

    # Run the Python script and capture its output
    output=$(python get_mean_std_length.py --dataset_name $dataset_name --input_csv "$output_csv_path/${dataset_name}_merged.csv")

    # Parse the output to extract the values
    eval "$output"

    # Use the extracted values in subsequent commands
    echo "Dataset Mean: $dataset_mean"
    echo "Dataset Std: $dataset_std"
    echo "Average Audio Length: $average_audio_length"

    eval "$(conda shell.bash hook)"
    conda activate ast

    cd ./ast/egs/esc50/

    # hard code both for now
    dataset_mean=-4.2677393
    dataset_std=4.5689974
    # audio_length=
    label_map="./ast/egs/esc50/data/${dataset_name}.json"

    if [ "$only_synthetic" = True ]; then
        train_csv="$output_csv_path/${dataset_name}_merged_synthetic.csv"
    else
        train_csv="$output_csv_path/${dataset_name}_merged.csv"
    fi

    fold_wise_eval="False"

    if [ "$use_ast" = True ]; then
      # finally train your AST
      if [ "$supcon" = False ]; then
          sh run_esc.sh $dataset_name $dataset_mean $dataset_std $average_audio_length $label_map $train_csv $valid_csv $test_csv $fold_wise_eval
      else
          sh run_esc_supcon.sh $dataset_name $dataset_mean $dataset_std $average_audio_length $label_map $train_csv $valid_csv $test_csv $fold_wise_eval $supcon_csv
      fi
    else
      conda activate clap
      cd /../../../aaai_2025/
      sh run_linear_probe.sh $train_csv $valid_csv $test_csv
      #Sonal enter code
    fi
      
else
    train_csv=$input_train_file

    # generate the label map for AST training
    python generate_label_map.py --dataset_name $dataset_name --input_csv $train_csv --output_json "./ast/egs/esc50/data/${dataset_name}.json" --output_csv_path ./ast/egs/esc50/data/

    output=$(python get_mean_std_length.py --dataset_name $dataset_name --input_csv "$train_csv")

    # Parse the output to extract the values
    eval "$output"

    # Use the extracted values in subsequent commands
    echo "Dataset Mean: $dataset_mean"
    echo "Dataset Std: $dataset_std"
    echo "Average Audio Length: $average_audio_length"

    eval "$(conda shell.bash hook)"
    conda activate ast

    cd ./ast/egs/esc50/

    dataset_mean=-4.2677393
    dataset_std=4.5689974
    # audio_length=
    label_map="./ast/egs/esc50/data/${dataset_name}.json"

    fold_wise_eval="False"

    sh run_esc.sh $dataset_name $dataset_mean $dataset_std $average_audio_length $label_map $train_csv $valid_csv $test_csv $fold_wise_eval
fi





