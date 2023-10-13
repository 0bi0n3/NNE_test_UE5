
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the audio file
    audio_path = 'test1.wav'
    y, sr = librosa.load(audio_path)

    # Display the waveform
    plt.figure(figsize=(12, 4))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Display the spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.stft(y)  # STFT of y
    db = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')

    # Print the sampling rate
    print(f'Sampling rate: {sr} Hz')

    # Show all plots
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
