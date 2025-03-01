import pandas as pd
import os
import sys

def split_csv(input_csv, k):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Calculate the number of rows per part
    num_rows = len(df)
    rows_per_part = num_rows // k
    remainder = num_rows % k
    
    start_row = 0
    
    # Extract the base name and directory from the input file path
    base_name = os.path.basename(input_csv)
    dir_name = os.path.dirname(input_csv)
    name, ext = os.path.splitext(base_name)
    
    for i in range(k):
        # Determine the end row for this part
        end_row = start_row + rows_per_part + (1 if i < remainder else 0)
        
        # Slice the DataFrame to get the current part
        part_df = df.iloc[start_row:end_row]
        
        # Construct the file name for the current part
        part_file_name = f"{name}_{i}{ext}"
        part_file_path = os.path.join(dir_name, part_file_name)
        
        # Save the current part to a new CSV file
        print(part_file_path)
        part_df.to_csv(part_file_path, index=False)
        
        # Update the start row for the next part
        start_row = end_row

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create JSON for label mapping')
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV', required=True)
    parser.add_argument('--num', type=int, help='Path to the input CSV', required=True)
    args = parser.parse_args()
    split_csv(args.input_csv, args.num)