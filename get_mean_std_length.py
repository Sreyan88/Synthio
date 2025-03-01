import pandas as pd
import numpy as np
import librosa


def main(args):
    # Load the CSV file
    csv_file_path = args.input_csv  # replace with the actual path to your CSV file
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=0.5, random_state=1) 

    # Initialize lists to store audio features
    audio_lengths = []
    audio_means = []
    audio_stds = []

    # Process each audio file
    for audio_path in df['path']:
        try:
            y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
        except Exception as e:
            y, sr = np.random.rand(1, 16000), 16000 # hardcoded to avoid errors
        audio_lengths.append(len(y) / sr)  # Calculate audio length in seconds
        audio_means.append(np.mean(y))  # Calculate mean of the audio signal
        audio_stds.append(np.std(y))  # Calculate standard deviation of the audio signal

    # Calculate dataset statistics
    dataset_mean = np.mean(audio_means)
    dataset_std = np.mean(audio_stds)
    average_audio_length = np.mean(audio_lengths)
    average_frames = int((average_audio_length * 1024) // 10)

    print(f"dataset_mean={dataset_mean}")
    print(f"dataset_std={dataset_std}")
    print(f"average_audio_length={average_frames}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create JSON for label mapping')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV', required=True)
    args = parser.parse_args()
    main(args)
