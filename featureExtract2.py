# Step 1: Load and Preprocess Data
# create a function that will simply get all the 
# file paths for the audio files in the specified folder.

import os

def load_file_paths(folder_path):
    file_paths = []

    for dirname, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            file_paths.append(file_path)
    
    return file_paths

# Step 2: Load Audio Files
# modify the function to load audio files and create Mel Spectrograms.

import librosa
import numpy as np

def load_and_extract_features(file_paths, sample_rate=48000):
    mel_specs = []

    for file_path in file_paths:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate)
        mel_specs.append(mel_spec)

    return np.array(mel_specs)

# Step 3: Save the Unlabeled Test Set
# save the Mel Spectrograms into a file.
def save_test_set(X, file_name):
    np.savez(file_name, X=X)

# Main Function to Execute Everything
def main(folder_path):
    file_paths = load_file_paths(folder_path)
    mel_specs = load_and_extract_features(file_paths)
    
    # Save the test data
    save_test_set(mel_specs, 'unlabeled_test_set.npz')

if __name__ == '__main__':
    main('your/folder/path')
