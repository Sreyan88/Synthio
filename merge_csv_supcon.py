import os
import sys
import pandas as pd

def merge_csv_files(output_csv_path, dataset_name, k, clap_filter, dpo):
    # List to store DataFrames
    dataframes = []
    
    # Iterate over the range 0 to K to read the CSV files
    for num_process in range(k):
        if dpo == "True":
            print("Merging DPO CSVs.")
            file_name = f"{output_csv_path}{dataset_name}_dpo_{num_process}.csv"
        else:
            if clap_filter == "True":
                print("Merging Filtered CSVs.")
                file_name = f"{output_csv_path}{dataset_name}_merged_filtered_{num_process}.csv"
            else:
                print("Merging Normal CSVs.")
                file_name = f"{output_csv_path}{dataset_name}_merged_{num_process}.csv"

        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            dataframes.append(df)
        else:
            print(f"File {file_name} does not exist. Skipping.")
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the merged DataFrame to a new CSV file
    if dpo == "True":
        output_file = f"{output_csv_path}{dataset_name}_dpo_merged.csv"
    else:
        output_file = f"{output_csv_path}{dataset_name}_merged.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create JSON for label mapping')
    parser.add_argument('--output_csv_path', type=str, help='Path to the input CSV', required=True)
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--num', type=int, help='Path to the input CSV', required=True)
    parser.add_argument('--clap_filter', type=str, help='If clap filter was applied', required=True)
    parser.add_argument('--dpo', type=str, help='If clap filter was applied', required=False, default="False")
    args = parser.parse_args()

    # Call the function to merge CSV files
    merge_csv_files(args.output_csv_path, args.dataset_name, args.num, args.clap_filter, args.dpo)