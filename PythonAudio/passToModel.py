import numpy as np
from sklearn.preprocessing import StandardScaler
import subprocess

# Assuming OUTPUT_PATH is where the mel spectrograms are saved
OUTPUT_PATH = 'g:/GameDev/UWL/modelDataTest'

# Load Mel spectrograms
def load_saved_mel_spectrograms(output_path):
    mel_spectrograms = []
    for i in range(num_files):  # Replace num_files with the actual number of saved spectrograms
        mel_path = os.path.join(output_path, f"mel_spectrogram_{i}.npy")
        mel_spec = np.load(mel_path)
        mel_spectrograms.append(mel_spec)
    return np.stack(mel_spectrograms, axis=0)

# Load the saved Mel spectrograms
X_test = load_saved_mel_spectrograms(OUTPUT_PATH)

# Scale the data
scaler = StandardScaler()
b, c, h, w = X_test.shape
X_test = np.reshape(X_test, newshape=(b, -1))
X_test = scaler.transform(X_test)
X_test = np.reshape(X_test, newshape=(b, c, h, w))

# Assuming the_model is the trained model
predictions = the_model.predict(X_test)  # Replace 'the_model' with the actual model variable

# Convert predictions to emotions (assuming you have a function for that)
emotions = predictions_to_emotions(predictions)  # Implement this function

# Print or send this data to Unreal Engine
print(emotions)

# Example: Sending a command to Unreal Engine
# subprocess.run(["ue4_command", str(emotions)])

# OR: Writing to a log file that Unreal reads
with open("emotion_log.txt", "w") as log_file:
    log_file.write(str(emotions))
