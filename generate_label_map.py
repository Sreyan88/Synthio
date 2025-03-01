import pandas as pd
import json


def main(args):
    # Load the CSV file
    file_path = args.input_csv  # replace with the actual path to your CSV file
    df = pd.read_csv(file_path)

    # Generate a unique number for each unique label
    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}

    # Save the mapping to a JSON file
    json_file_path = args.output_json
    with open(json_file_path, 'w') as json_file:
        json.dump(label_mapping, json_file, indent=4)

    print(f"JSON file saved at {json_file_path}")

    random_string = "/m/07rwj"
    csv_data = []

    for label, idx in label_mapping.items():
        mid = f"{random_string}{str(idx).zfill(2)}"
        csv_data.append([idx, mid, label])

    # Convert the list to a DataFrame
    csv_df = pd.DataFrame(csv_data, columns=["index", "mid", "display_name"])

    # Save the DataFrame to a CSV file
    
    # csv_df.to_csv(csv_file_path, index=False, quotechar='"', quoting=1)
    csv_df['display_name'] = csv_df['display_name'].apply(lambda x: f'"{x}"')

    # Save the DataFrame to a CSV file without additional quoting
    csv_file_path = args.output_csv_path + args.dataset_name + '_class_labels_indices.csv'
    csv_df.to_csv(csv_file_path, index=False, quoting=3)


    print(f"CSV file saved at {csv_file_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create JSON for label mapping')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV', required=True)
    parser.add_argument('--output_json', type=str, help='Path to output JSON', required=True)
    parser.add_argument('--output_csv_path', type=str, help='Path to output CSV', required=True)
    args = parser.parse_args()
    main(args)