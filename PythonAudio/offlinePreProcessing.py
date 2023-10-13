import os
import numpy as np
import librosa
import soundfile as sf  # For writing audio files

# Global Variables
SAMPLE_RATE = 48000
DATA_PATH = 'g:/GameDev/UWL/NNETutorial - Copy/Saved/BouncedWavFiles/NNAudio'  # Input directory
OUTPUT_PATH = 'g:/GameDev/UWL/modelDataTest'  # Output directory

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):

    os.makedirs(OUTPUT_PATH)


# Function to load audio files
def load_audio_files(data_path):

    audio_files = []
    for root, dirs, files in os.walk(data_path):
    
        for file in files:
        
            if file.endswith('.wav'):
            
                audio_files.append(os.path.join(root, file))
                    
    return audio_files


# Function to preprocess audio
def preprocess_audio(audio_path):

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=3, offset=0.5)
    padded_audio = np.zeros((SAMPLE_RATE * 3,))
    padded_audio[:len(audio)] = audio
    return padded_audio


# Function to generate mel spectrograms
def get_mel_spectrogram(audio):

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=1024, win_length=512,
        hop_length=256, n_mels=128, fmax=SAMPLE_RATE // 2
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


# Main function
def main():

    audio_files = load_audio_files(DATA_PATH)
    mel_spectrograms = []

    for audio_file in audio_files:
    
        # Preprocess the audio
        audio = preprocess_audio(audio_file)
        
        # Generate mel spectrogram
        mel_spec = get_mel_spectrogram(audio)
        mel_spectrograms.append(mel_spec)

        # Write preprocessed audio to file
        output_filename = 'processed_' + os.path.basename(audio_file)
        output_file_path = os.path.join(OUTPUT_PATH, output_filename)
        sf.write(output_file_path, audio, SAMPLE_RATE)
    

    # Save the mel spectrograms
    for i, mel_spec in enumerate(mel_spectrograms):
    
        output_file_path = os.path.join(OUTPUT_PATH, f"mel_spectrogram_{i}.npy")
        np.save(output_file_path, mel_spec)
    


# Call main function to run the script
if __name__ == '__main__':
    main()

