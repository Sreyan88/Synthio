import argparse
import json
import numpy as np
import csv

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
    return index_lookup

def assign_equal_weight(data, index_dict):
    sample_weight = np.ones(len(data))
    return sample_weight

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_path", type=str, required=True, help="the root path of data json file")
parser.add_argument("--label_csv", type=str, required=True, help="the path of label csv file")

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    label_csv = args.label_csv

    index_dict = make_index_dict(label_csv)

    with open(data_path, 'r', encoding='utf8') as fp:
        data = json.load(fp)['data']

    sample_weight = assign_equal_weight(data, index_dict)
    np.savetxt(data_path[:-5]+'_weight.csv', sample_weight, delimiter=',')
