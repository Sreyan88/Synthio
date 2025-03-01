import json
import os
import requests

import openai
import pandas as pd

import base64
import requests
from tqdm.auto import tqdm
import sys
import ast
import random

random.seed(42)

extraction_prompt = "I will provide you with a caption of an audio that describes the events that take place in the audio. Additionally, I will also provide you with a label for the audio. Extract the phrases that correspond to 1) the distinctive events in the caption and 2) the broader background scene in the events or audio that are taking place. If you think there is no information for either of the keys, leave them empty. Return a JSON with key 2 keys, one as named as events and the other as named as scenes, where the values of these keys correspond to a comma-separated pythonoic list where each item in the list is a string corresponding to the extracted phrases. Please ignore any phrase that (exactly or semantically) correponds to the label of the audio. \n\nHere is the caption:{}\nHere is the label:{}"

prompt = "I will provide you with a caption of an audio that describes the events that take place in the audio. Additionally, I will also provide you with a label for the audio that describes the audio in an abstract fashion and mentions the broader scene or event that I need to teach an audio model about from the audio, i.e., the audio will be part of the training set for training the model. I would like you to generate 5 new captions from the given caption. I will use these captions to generate new audios that can augment my training set. Generate the new captions with the following requirements:\n\n1. The captions need to include new events and new contexts from the original caption. Additionally, each caption should have a diverse added events or contexts. \n2. Only add new events and context by understanding the broader context of the occurrence of the audio and the target label. Do not add random events or contexts.\n3. The new caption should be not more than 20-25 words.\n4. However, after adding new events or contexts, the caption still needs to obey the event label, i.e., the new caption may not lead to an audio generation that defies the audio label.\n\nHere is the caption:{}\n\nHere is the label:{}. \n\nOutput a JSON with the key as the original caption and a value as the list of comma separated new captions. Only output the JSON and nothing else."

prompt_filtered = "I will provide you with a label for an audio. The label generally describes the audio in an abstract fashion and mentions the broader scene or event that I need to teach an audio model about from the audio, i.e., the audio and its label is part of the training set for training an audio classification model. I will also provide you with the domain of the audio which will help you identify the true sound conveyed in the label. I would like you to generate 5 new captions that describe the event or source in the label in diverse fashions. I will use these captions to generate new audios that can augment my training set. Generate the new captions with the following requirements:\n\n1. Each caption should have a diverse added events (beyond the event of the original label) and contexts. \n2. Only add new events and context by understanding the broader context of the occurrence of the audio and the target label. For adding events and contexts, please follow the next requirement. \n3. I will also provide you with a list of events and backgrounds. You should try such that the new captions you generate for the label have a portion of events and scenes similar to the events and background scenes that are given, i.e., you should try to describe a scene for the caption with the events and backgrounds provided to you in the given lists but you are encouraged to have a small portion of novel events and backgrounds beyond the ones given. \n3. The new caption should be not more than 20-25 words. \n5. However, after all these constraints and adding new events or contexts, the caption still needs to obey the event label, i.e., the new caption may not lead to an audio generation that defies the audio label.\n6. Finally, use the original label as a phrase in your caption.\n\nHere is the label:{}. \n\nHere is the domain of the audio:{}. \nHere is the list of events:{}\nHere is the list of backgrounds:{}\n\nOutput a JSON with the key as the original caption and a value as the list of comma separated new captions. Only output the JSON and nothing else."

prompt_filtered_rewrite = "I will provide you with a caption for an audio. The label generally describes the audio in an abstract fashion and mentions the broader scene or event that I need to teach an audio model about from the audio, i.e., the audio and its label is part of the training set for training an audio classification model. I will also provide you with the domain of the audio which will help you identify the true sound conveyed in the label. I need you to rewrite the caption for me according to this set of rules: \n1. I will provide you with a list of events and backgrounds. You should try such that the rewritten caption have events and scenes beyond the events and background scenes that are given, i.e., you should try to describe a scene for the label with  events and backgrounds different from the ones given. If you think the caption is diverse enough and does not have much repetitive features (compared to the given features) or if rewritten will divert from the label, do not change it. \n2. After re-writing, the caption should still obey the audio event label.\n\nHere is the label:{}. \n\nHere is the domain of the audio:{}. \nHere is the list of events:{}\nHere is the list of backgrounds:{}\n\nJust output the rewritten caption and nothing else. Output 'None' if you did not rewrite."

prompt_supcon = "I will provide you with a label for an audio. The label generally describes the audio in an abstract fashion and mentions the broader scene or event that I need to teach an audio model about from the audio, i.e., the audio and its label is part of the training set for training an audio classification model. I will also provide you with the domain of the audio which will help you identify the true sound conveyed in the label. I would like you to generate 5 new captions that describe the event or source in the label in diverse fashions. I will use these captions to generate new audios that can augment my training set. Generate the new captions with the following requirements:\n\n1. All the captions need to include new and diverse events and contexts beyond the actual event conveyed by the label. \n2. Only add new events and context by understanding the broader context of the occurrence of the audio and the target label. Do not add random events or contexts.\n3. The new caption should be not more than 20-25 words.\n4. However, after all these constraints and adding new events or contexts, the caption still needs to obey the event conveyed by the original label, i.e., the new caption may not lead to an audio generation that defies the audio label.\n6. Finally, use the original label as a phrase in your caption.\n\nHere is the label:{}. \n\nHere is the domain of the audio:{}. Output a JSON with the key as the original label and a value as the list of comma separated new captions. Only output the JSON and nothing else."

prompt_multi_label = "I will provide you with 5 audio labels. The labels generally describes the audio in an abstract fashion and mentions the broader scene or event that I need to teach an audio model about from the audio, i.e., the audio and its labels is part of the training set for training a multi-label audio classification model. I will also provide you with the domain of the audio which will help you identify the true sounds conveyed in the labels. I would like you to generate 5 new captions that describe the event or source in 2-4 labels in diverse fashions. I will use these captions to generate new audios that can augment my training set. Generate the new captions with the following requirements:\n\n1. All the captions need to include diverse contexts with the events in the labels taking place in the scene. \n2. Randomly select 2-4 labels (among the 5 labels provided) for making each caption. \n3. Only add new contexts by understanding the broader context of the occurrence of the audio and the target labels. Do not add random contexts.\n4. The new caption should be not more than 20-25 words.\n5. However, after all these constraints and adding new events or contexts, the caption still needs to obey the event conveyed by the original label, i.e., the new caption may not lead to an audio generation that defies the audio label.\n6. Finally, use the original label in your caption.\n\nHere is the label:{}. \n\nHere is the domain of the audio:{}. Output a JSON with 5 key-value pairs. Each key should be the list of original labels used to generate the caption and the value as the new caption. Only output the JSON and nothing else."

# input your OpenAI key
openai.api_key = ''


def get_flattened_caption_features_by_label(dataframe, label, column):
    # Filter the DataFrame for the specified label and non-null gpt_captions
    filtered_df = dataframe[(dataframe['label'] == label) & (dataframe[column] != 'None')]
    
    # Initialize an empty list to store the flattened captions
    flattened_captions = []
    
    # Iterate through the filtered DataFrame and extend the flattened_captions list
    for captions in filtered_df[column]:
        captions_list = ast.literal_eval(captions)  # Convert string representation of list to actual list
        flattened_captions.extend(captions_list)
    
    return flattened_captions


def select_half_items(input_list):
    # Determine the number of items to select (50% of the input list)
    num_items_to_select = int(len(input_list) * 0.75)
    
    # Randomly select the specified number of items from the input list
    selected_items = random.sample(input_list, num_items_to_select)
    
    return selected_items

def prompt_gpt(prompt_input, json_output = True):

    response = openai.ChatCompletion.create(model='gpt-4o',
                                                messages=[
                                                            {
                                                            "role": "user",
                                                            "content": [
                                                                {
                                                                "type": "text",
                                                                "text": prompt_input
                                                                },
                                                                       ]
                                                            }
                                                        ],
                                                temperature=0.7,
                                                max_tokens=4096,
                                                response_format={ 'type': 'json_object' }
                                                )
    return response

def generate_plain_captions(args):

    input_df = pd.read_csv(args.input_csv)
    # input_df['gpt_captions'] = ['None' for i in range(len(input_df))]

    for i,row in tqdm(input_df.iterrows()):

        try:
            if ":" in row['caption']:
                temp_caption = row['caption'].split(": ")[-1]
            else:
                temp_caption = row['caption']

            x = prompt.format(temp_caption, " ".join(row['label'].split("_"))) # row['caption'][len("Audio caption: "):]

            response = prompt_gpt(x)
            prediction = eval(response['choices'][0]['message']['content'].replace('\n',''))

            # input_df.iloc[i]['gpt_captions'] = prediction

            data = {
                    'generated_captions': list(prediction.values())[0],
                    'path': row['path']
                    }
            
            # print(data)
            
            with open(f'{args.input_csv}.plain.jsonl', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)

def generate_plain_captions_wo_captions(args):

    input_df = pd.read_csv(args.input_csv)
    # input_df['gpt_captions'] = ['None' for i in range(len(input_df))]

    for i,row in tqdm(input_df.iterrows()):

        try:

            x = prompt_supcon.format(" ".join(row['label'].split("_")), args.domain)

            response = prompt_gpt(x)
            prediction = eval(response['choices'][0]['message']['content'].replace('\n',''))

            # input_df.iloc[i]['gpt_captions'] = prediction

            data = {
                    'generated_captions': list(prediction.values())[0],
                    'path': row['path']
                    }
            
            # print(data)
            
            with open(f'{args.input_csv}.plain_wo_caption.jsonl', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)

def generate_plain_captions_wo_captions_multi_label(args):

    input_df = pd.read_csv(args.input_csv)
    # input_df['gpt_captions'] = ['None' for i in range(len(input_df))]
    all_labels = list(set(input_df['label'].tolist()))

    for i,row in tqdm(input_df.iterrows()):

        try:

            selected_labels = random.sample(all_labels, 5)

            x = prompt_multi_label.format(" ".join(selected_labels), args.domain)

            response = prompt_gpt(x)
            prediction = eval(response['choices'][0]['message']['content'].replace('\n',''))

            # input_df.iloc[i]['gpt_captions'] = prediction

            data = {
                    'generated_captions': list(prediction.values())[0],
                    'path': row['path']
                    }
            
            # print(data)
            
            with open(f'{args.input_csv}.plain_wo_caption_multilabel.jsonl', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)

def generate_filtered_captions(args):

    input_df = pd.read_csv(args.input_csv)


    for i,row in tqdm(input_df.iterrows()):

        try:

            events = select_half_items(get_flattened_caption_features_by_label(input_df, " ".join(row['label'].split("_")), "gpt_caption_features_events"))
            
            scenes = select_half_items(get_flattened_caption_features_by_label(input_df, " ".join(row['label'].split("_")), "gpt_caption_features_scenes"))

            x = prompt_filtered.format(" ".join(row['label'].split("_")), args.domain, events, scenes)

            response = prompt_gpt(x)
            prediction = eval(response['choices'][0]['message']['content'].replace('\n',''))

            # input_df.iloc[i]['gpt_captions'] = prediction

            data = {'generated_captions': list(prediction.values())[0],
                    'path': row['path']
                    }
            
            # print(data)
            
            with open(f'{args.input_csv}.filtered.jsonl', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)

def generate_filtered_captions_iteration_stage(args):

    input_df = pd.read_csv(args.input_csv)

    non_filteredout_df = args.input_csv[:-len('_filteredout_merged.csv')] + '_merged.csv'

    for i,row in tqdm(input_df.iterrows()):

        try:

            events = select_half_items(get_flattened_caption_features_by_label(non_filteredout_df, " ".join(row['label'].split("_")), "gpt_caption_features_events"))
            
            scenes = select_half_items(get_flattened_caption_features_by_label(non_filteredout_df, " ".join(row['label'].split("_")), "gpt_caption_features_scenes"))

            x = prompt_filtered_rewrite.format(" ".join(row['label'].split("_")), args.domain, events, scenes)

            response = prompt_gpt(x)
            prediction = eval(response['choices'][0]['message']['content'].replace('\n',''))

            # input_df.iloc[i]['gpt_captions'] = prediction

            data = {'generated_caption': prediction,
                    'path': row['path']
                    }
        
            # print(data)
            
            with open(f'{args.input_csv}.iter.jsonl', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)

def get_caption_features(args):

    if args.iteration_stage == "True":
        # if iteration stage, take features from non filtered part (above threshold)
        args.input_csv = args.input_csv[:-len('_filteredout_merged.csv')] + '_merged.csv'

    input_df = pd.read_csv(args.input_csv)
    # input_df['gpt_captions'] = ['None' for i in range(len(input_df))]

    for i,row in tqdm(input_df.iterrows()):

        try:

            x = extraction_prompt.format(row['caption'][len("Audio caption: "):], row['label'])

            response = prompt_gpt(x)
            prediction = eval(response['choices'][0]['message']['content'].replace('\n',''))

            # input_df.iloc[i]['gpt_captions'] = prediction

            data = {"events": prediction["events"],
                    "scenes": prediction["scenes"],
                    "path": row["path"],
                    "caption": row["caption"]
                    }
            
            # print(data)
            
            with open(f'{args.input_csv}.features.jsonl', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)


def generate_supcon_captions(args):

    input_df = pd.read_csv(args.input_csv)

    for i,row in tqdm(input_df.iterrows()):

        try:

            x = prompt_supcon.format(" ".join(row['label'].split("_")), args.domain)

            response = prompt_gpt(x)

            prediction = eval(response['choices'][0]['message']['content'].replace('\n',''))

            data = {'generated_captions': list(prediction.values())[0],
                    'path': row['path']
                    }
            
            # print(data)
            
            with open(f'{args.input_csv}.supcon.jsonl', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)

def merge_generated_caption_features(args):

    input_df = pd.read_csv(args.input_csv)
    input_df['gpt_caption_features_events'] = ['None' for i in range(len(input_df))]
    input_df['gpt_caption_features_scenes'] = ['None' for i in range(len(input_df))]

    # Path to your JSONLines file
    file_path = f'{args.input_csv}.features.jsonl'

    # Reading the JSONLines file
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            obj = json.loads(line)
            input_df.loc[input_df['path'] == obj['path'], 'gpt_caption_features_events'] = str(obj['events'])
            input_df.loc[input_df['path'] == obj['path'], 'gpt_caption_features_scenes'] = str(obj['scenes'])

    input_df.to_csv(args.input_csv, index=False)

def merge_generated_captions(args):

    input_df = pd.read_csv(args.input_csv)
    input_df['gpt_captions'] = ['None' for i in range(len(input_df))]

    # Path to your JSONLines file
    if args.supcon == "True":
        file_path = f'{args.input_csv}.supcon.jsonl'
        column_name = 'gpt_captions_supcon'
    elif args.multi_label == "True":
        file_path = f'{args.input_csv}.plain_wo_caption_multilabel.jsonl'
        column_name = 'gpt_captions'
    else:
        if args.plain_caption == "True":
            if args.plain_wo_caption == "True":
                file_path = f'{args.input_csv}.plain_wo_caption.jsonl'
            else:
                file_path = f'{args.input_csv}.plain.jsonl'
        else:
            file_path = f'{args.input_csv}.filtered.jsonl'
        column_name = 'gpt_captions'

    if args.multi_label == "True":
        with open(file_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                # complete later
    else:
        # Reading the JSONLines file
        with open(file_path, 'r') as file:
            for line in file:
                # Parse each line as a JSON object
                obj = json.loads(line)
                input_df.loc[input_df['path'] == obj['path'], column_name] = str(obj['generated_captions'])

    input_df.to_csv(args.input_csv, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate GPT Captions')
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV', required=True)
    parser.add_argument('--plain_caption', type=str, help='Whether to generate plain caption or with proposed algorithm', required=True)
    parser.add_argument('--supcon', type=str, help='Whether you are generating captions for supervised contrastive learning', required=True)
    parser.add_argument('--domain', type=str, help='The domain in which the dataset belongs.', required=True)
    parser.add_argument('--plain_wo_caption', type=str, help='If you want to generate prompts without the use of caption in the algorithm', required=True)
    parser.add_argument('--multi_label', type=str, help='If task is multi-label', default="False", required=False)
    parser.add_argument('--iteration_stage', type=str, help='If it is the iterative refinement stage', default="False", required=False)


    args = parser.parse_args()

    if args.supcon == "True":
        if not os.path.exists(f'{args.input_csv}.supcon.jsonl'):
            generate_supcon_captions(args)
        merge_generated_captions(args)
    elif args.multi_label == "True":
        if not os.path.exists(f'{args.input_csv}.plain_wo_caption_multilabel.jsonl'):
            generate_plain_captions_wo_captions_multi_label(args)
            merge_generated_captions(args)
        else:
             merge_generated_captions(args)
    elif args.iteration_stage == "True":
        get_caption_features(args)
        merge_generated_caption_features(args)
        generate_filtered_captions_iteration_stage(args)
    else:
        if args.plain_caption == "False":
            if not os.path.exists(f'{args.input_csv}.filtered.jsonl'):
                if not os.path.exists(f'{args.input_csv}.features.jsonl'):
                    get_caption_features(args)
                    merge_generated_caption_features(args)
                else:
                    merge_generated_caption_features(args)
                generate_filtered_captions(args)
                merge_generated_captions(args)
            else:
                merge_generated_captions(args)
        elif args.plain_caption == "True":
            if args.plain_wo_caption == "False":
                if not os.path.exists(f'{args.input_csv}.plain.jsonl'):
                    generate_plain_captions(args)
                    merge_generated_captions(args)
                else:
                    merge_generated_captions(args)
            elif args.plain_wo_caption == "True":
                if not os.path.exists(f'{args.input_csv}.plain_wo_caption.jsonl'):
                    generate_plain_captions_wo_captions(args)
                    merge_generated_captions(args)
                else:
                    merge_generated_captions(args)

