#!/bin/bash

set -x

model=ast
dataset=$1
imagenetpretrain=True
audiosetpretrain=True
bal=none

if [ "$audiosetpretrain" == "True" ]; then
  lr=1e-5
else
  lr=1e-4
fi

freqm=0  #48  #24
timem=0  #192    #96
mixup=0
epoch=50
batch_size=48
fstride=10
tstride=10

dataset_mean=$2
dataset_std=$3
audio_length=$4
noise=False

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

label_map=$5
train_csv=$6
valid_csv=$7
test_csv=$8
fold_wise_eval=$9

echo "${fold_wise_eval}"

base_exp_dir=./exp/test-${dataset}-f${fstride}-t${tstride}-imp${imagenetpretrain}-asp${audiosetpretrain}-b${batch_size}-lr${lr}

python ./prep_esc50.py --label_map_json $label_map --train_csv $train_csv --valid_csv $valid_csv --test_csv $test_csv --fold_wise_eval $fold_wise_eval --dataset_name $dataset

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  rm -rf $base_exp_dir
fi

mkdir -p $base_exp_dir

if [ "$fold_wise_eval" == "True" ]; then
  for ((fold=1; fold<=5; fold++)); do
    echo 'now process fold'${fold}

    exp_dir=${base_exp_dir}/fold${fold}

    tr_data=./data/datafiles/esc_train_data_${fold}.json
    te_data=./data/datafiles/esc_eval_data_${fold}.json

    CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
      --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
      --label-csv ./data/esc_class_labels_indices.csv --n_class 10 \
      --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
      --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
      --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
      --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
      --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise}

    python ./get_esc_result.py --exp_path ${base_exp_dir}
  done
else
  exp_dir=${base_exp_dir}/fold

  tr_data=/fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_${dataset}/${dataset}_train_data.json
  tv_data=/fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_${dataset}/${dataset}_val_data.json
  te_data=/fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_${dataset}/${dataset}_test_data.json

  csv_file="./data/${dataset}_class_labels_indices.csv"
  line_count=$(tail -n +2 "$csv_file" | wc -l)

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
    --data-train ${tr_data} --data-val ${te_data} --data-eval ${te_data} --exp-dir $exp_dir \
    --label-csv ./data/${dataset}_class_labels_indices.csv --n_class ${line_count} \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
    --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
    --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
    --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise}

  python ./get_esc_result.py --exp_path ${base_exp_dir}
fi
