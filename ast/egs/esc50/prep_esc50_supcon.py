import numpy as np
import json
import os
import zipfile
import wget
import pandas as pd

def main(args):

    # with open(args.label_map_json,'r') as f:
    #     label_map = json.load(f)

    label_set = np.loadtxt('./data/' + args.dataset_name + '_class_labels_indices.csv', delimiter=',', dtype='str')
    label_map = {}
    for i in range(1, len(label_set)):
        label_map[eval(label_set[i][2])] = label_set[i][0] # display_name: index
    print(label_map)

    # fix bug: generate an empty directory to save json files
    if os.path.exists('./data/datafiles' + "_" + args.dataset_name) == False:
        os.mkdir('./data/datafiles' + "_" + args.dataset_name)

    args.fold_wise_eval = False
    # print(args.fold_wise_eval)
    if args.fold_wise_eval:
        for fold in [1,2,3,4,5]:
            base_path = "./data/ESC-50-master/audio_16k/"
            meta = np.loadtxt('./data/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
            train_wav_list = []
            eval_wav_list = []
            for i in range(0, len(meta)):
                cur_label = label_map[meta[i][3]]
                cur_path = meta[i][0]
                cur_fold = int(meta[i][1])
                # /m/07rwj is just a dummy prefix
                cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
                if cur_fold == fold:
                    eval_wav_list.append(cur_dict)
                else:
                    train_wav_list.append(cur_dict)

            print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

            with open('./data/datafiles/esc_train_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': train_wav_list}, f, indent=1)

            with open('./data/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': eval_wav_list}, f, indent=1)

    else:

        train_wav_list = []
        
        train = pd.read_csv(args.train_csv)

        for i,row in train.iterrows():
            cur_label = label_map[row['label']]
            cur_dict = {"wav": eval(row['path_new']), "labels": '/m/07rwj'+cur_label.zfill(2)}
            train_wav_list.append(cur_dict)


        with open('/fs/nexus-projects/brain_project/aaai_2025/ast/egs/esc50/data/datafiles_' + args.dataset_name + '/' + args.dataset_name + '_train_data_supcon' +'.json', 'w') as f:
            json.dump({'data': train_wav_list}, f, indent=1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create JSON for label mapping')
    parser.add_argument('--label_map_json', type=str, help='Path to label map JSON', required=False)
    parser.add_argument('--train_csv', type=str, help='Path to train CSV', required=False)
    parser.add_argument('--fold_wise_eval', type=bool, help='If evaluation strategy is multi-fold', required=True, default=False)
    parser.add_argument('--dataset_name', type=str, help='Dataset name', required=True, default=False)
    args = parser.parse_args()
    main(args)