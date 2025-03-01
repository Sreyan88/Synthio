import pandas as pd
from msclap import CLAP
import torch
import torch.nn.functional as F


def main(args):

    input_csv = f"{args.input_csv_path}{args.dataset_name}_merged_{args.iter}.csv"

    # print('File Read.')

    input_df = pd.read_csv(input_csv)
    clap_model = CLAP(version = '2023', use_cuda=True)

    new_audios_list = []
    new_labels_list = []
    new_dataset_list = []
    new_split_name_list = []
    new_caption_list = []

    only_synthetic_audios_list = []
    only_synthetic_labels_list = []
    only_synthetic_dataset_list = []
    only_synthetic_split_name_list = []
    only_synthetic_caption_list = []

    filteredout_audios_list = []
    filteredout_labels_list = []
    filteredout_dataset_list = []
    filteredout_split_name_list = []
    filteredout_caption_list = []


    for i,row in input_df.iterrows():


        if row['split_name'] == 'synthetic_augs':

            try:
                text_embeddings = clap_model.get_text_embeddings(["Sound of a " + " ".join(row['label'].split("_"))])[0]
                audio_embeddings = clap_model.get_audio_embeddings([row['path']])[0]
                
                audio_embeddings = audio_embeddings / torch.norm(audio_embeddings)
                text_embeddings = text_embeddings / torch.norm(text_embeddings)

                logit_scale = clap_model.clap.logit_scale.exp()
            
                similarity = logit_scale * (text_embeddings @ audio_embeddings.T)
                similarity = torch.sigmoid(similarity)

                if float(similarity.T.detach().cpu()) >= float(args.clap_threshold):
                    new_audios_list.append(row['path'])
                    new_labels_list.append(row['label'])
                    new_dataset_list.append(row['dataset'])
                    new_split_name_list.append(row['split_name'])
                    if args.use_label == "False":
                        new_caption_list.append(row['caption'])

                    # only synthetic
                    only_synthetic_audios_list.append(row['path'])
                    only_synthetic_labels_list.append(row['label'])
                    only_synthetic_dataset_list.append(row['dataset'])
                    only_synthetic_split_name_list.append(row['split_name'])
                    if args.use_label == "False":
                        only_synthetic_caption_list.append(row['caption'])

                else:
                    filteredout_audios_list.append(row['path'])
                    filteredout_labels_list.append(row['label'])
                    filteredout_dataset_list.append(row['dataset'])
                    filteredout_split_name_list.append(row['split_name'])
                    if args.use_label == "False":
                        filteredout_caption_list.append(row['caption'])

            except Exception as e:
                print(e)
                pass
        else:
            new_audios_list.append(row['path'])
            new_labels_list.append(row['label'])
            new_dataset_list.append(row['dataset'])
            new_split_name_list.append(row['split_name'])
            if args.use_label == "False":
                new_caption_list.append(row['caption'])

    # save filtered csv        
    filtered_df = pd.DataFrame()
    filtered_df['path'] = new_audios_list
    filtered_df['label'] = new_labels_list
    filtered_df['dataset'] = new_dataset_list
    filtered_df['split_name'] = new_split_name_list
    if args.use_label == "False":
        filtered_df['caption'] = new_caption_list


    output_csv = f"{args.input_csv_path}{args.dataset_name}_merged_filtered_{args.iter}.csv"

    filtered_df.to_csv(output_csv, index = False)

    # -------------------- #
    # save synthetic only csv
    only_synthetic_df = pd.DataFrame()
    only_synthetic_df['path'] = only_synthetic_audios_list
    only_synthetic_df['label'] = only_synthetic_labels_list
    only_synthetic_df['dataset'] = only_synthetic_dataset_list
    only_synthetic_df['split_name'] = only_synthetic_split_name_list
    if args.use_label == "False":
        only_synthetic_df['caption'] = only_synthetic_caption_list

    only_synthetic_output_csv = f"{args.input_csv_path}{args.dataset_name}_synthetic_{args.iter}.csv"

    only_synthetic_df.to_csv(only_synthetic_output_csv, index = False)


    # -------------------- #
    # save filtered out csv
    filteredout_df = pd.DataFrame()
    filteredout_df['path'] = filteredout_audios_list
    filteredout_df['label'] = filteredout_labels_list
    filteredout_df['dataset'] = filteredout_dataset_list
    filteredout_df['split_name'] = filteredout_split_name_list
    if args.use_label == "False":
        filteredout_df['caption'] = filteredout_caption_list


    filteredout_output_csv = f"{args.input_csv_path}{args.dataset_name}_filteredout_{args.iter}.csv"

    filteredout_df.to_csv(filteredout_output_csv, index = False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Filter Audios according to CLAP thereshold')
    parser.add_argument('--input_csv_path', type=str, help='Path to input csv', required=True)
    parser.add_argument('--iter', type=int, help='Path to output csv', required=True)
    parser.add_argument('--clap_threshold', type=float, help='CLAP threhold for filtering', required=True)
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--use_label', type=str, help='Use label or caption', required=True)

    args = parser.parse_args()
    main(args)