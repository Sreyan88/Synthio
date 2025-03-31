# python -W ignore ../../src/run.py --model ast --dataset nsynth --data-train /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/n
# synth_train_data.json --data-val /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/nsynth_test_data.json --data-eval /fs/nexus-pr
# ojects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/nsynth_test_data.json --exp-dir ./exp/test-nsynth-f10-t10-impTrue-aspTrue-b48-lr1e-5/fold -
# -label-csv ./data/nsynth_class_labels_indices.csv --n_class 11 --lr 1e-5 --n-epochs 25 --batch-size 48 --save_model False --freqm 24 --timem 96 --mixup 0 --ba
# l none --tstride 10 --fstride 10 --imagenet_pretrain True --audioset_pretrain True --metrics acc --loss CE --warmup False --lrscheduler_start 5 --lrscheduler_
# step 1 --lrscheduler_decay 0.85 --dataset_mean -4.2677393 --dataset_std 4.5689974 --audio_length 409 --noise False

python -W ignore ../../src/run_multi_conpos.py --model ast --dataset nsynth --data-train /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/n
synth_train_data.json --data-val /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/nsynth_test_data.json --data-eval /fs/nexus-pr
ojects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/nsynth_test_data.json --exp-dir ./exp/test-nsynth-f10-t10-impTrue-aspTrue-b48-lr1e-5/fold -
-label-csv ./data/nsynth_class_labels_indices.csv --n_class 11 --lr 1e-5 --n-epochs 25 --batch-size 48 --save_model False --freqm 24 --timem 96 --mixup 0 --ba
l none --tstride 10 --fstride 10 --imagenet_pretrain True --audioset_pretrain True --metrics acc --loss CE --warmup False --lrscheduler_start 5 --lrscheduler_
step 1 --lrscheduler_decay 0.85 --dataset_mean -4.2677393 --dataset_std 4.5689974 --audio_length 409 --noise False

CUDA_VISIBLE_DEVICES=0 python -W ignore ../../src/run_multi_conpos.py --model ast --dataset nsynth --data-train /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/nsynth_train_data.json --data-val /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/nsynth_test_data.json --data-eval /fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_nsynth/nsynth_test_data.json --exp-dir ./exp/test-nsynth-f10-t10-impTrue-aspTrue-b48-lr1e-5/fold --label-csv ./data/nsynth_class_labels_indices.csv --n_class 11 --lr 1e-5 --n-epochs 25 --batch-size 32 --save_model False --freqm 24 --timem 96 --mixup 0 --bal none --tstride 10 --fstride 10 --imagenet_pretrain True --audioset_pretrain True --metrics acc --loss CE --warmup False --lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --dataset_mean -4.2677393 --dataset_std 4.5689974 --audio_length 409 --noise False