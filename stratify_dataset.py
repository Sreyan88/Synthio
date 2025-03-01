import pandas as pd
import sys
from sklearn.model_selection import train_test_split

import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def stratified_sample(input_csv, n, output_csv, dataset_name, multi_label):

    # Read the CSV file
    df = pd.read_csv(input_csv)

    if 'dataset' not in df.columns:
        df['dataset'] = dataset_name

    if 'split_name' not in df.columns:
        df['split_name'] = 'train'

    if multi_label == 'True':
        # randomly sample n samples from the dataset if multi-label
        sampled_df = df.sample(n, random_state=1).reset_index(drop=True)

        # Write the sampled dataframe to a new CSV file
        sampled_df.to_csv(output_csv, index=False)
        
        print(f"Sampled dataset saved to {output_csv}")

    else:
        # Get the total number of samples per label
        label_counts = df['label'].value_counts()
        
        # Calculate the sampling fraction per label
        sample_frac = n / len(df)
        
        # Ensure that at least one sample per label is taken
        min_samples = {label: max(1, int(count * sample_frac)) for label, count in label_counts.items()}
        
        # Perform stratified sampling
        sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min_samples[x.name], random_state=1))
        
        # Shuffle the resultant dataframe to ensure randomness
        sampled_df = sampled_df.sample(frac=1, random_state=1).reset_index(drop=True)
        
        # Write the sampled dataframe to a new CSV file
        sampled_df.to_csv(output_csv, index=False)
        
        print(f"Sampled dataset saved to {output_csv}")





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Do stratified sampling of csv')
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV', required=True)
    parser.add_argument('--num_samples', type=int, help='Path to the dataset', required=True)
    parser.add_argument('--output_csv', type=str, help='Path to the input CSV', required=True)
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--multi_label', type=str, help='If multi label dataset', required=False)
    args = parser.parse_args()

    # Call the function to merge CSV files
    stratified_sample(args.input_csv, args.num_samples, args.output_csv, args.dataset_name, args.multi_label)