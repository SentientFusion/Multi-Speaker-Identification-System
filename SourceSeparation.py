import torch
import torchaudio
import soundfile as sf
import os
import tkinter as tk
from tkinter import filedialog
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Create output directory if not exists
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GUI File Picker
root = tk.Tk()
root.withdraw()  # Hide the root window
audio_path = filedialog.askopenfilename(title="Select the mixed audio file",
                                        filetypes=[("Audio Files", "*.wav;*.mp3;*.flac")])
if not audio_path:
    print("No file selected. Exiting.")
    exit()

# Load pre-trained Hybrid Demucs model
print("Loading speech separation model...")
model = get_model(name="htdemucs")  # Hybrid Transformer-based Demucs
model.eval()  # Set model to evaluation mode

# Load audio
waveform, sample_rate = torchaudio.load(audio_path)

# Ensure 2-channel (stereo) input (Fix)
if waveform.shape[0] == 1:
    waveform = waveform.repeat(2, 1)  # Duplicate mono channel to stereo

# Apply the model to separate speech
print("Separating speech sources...")
with torch.no_grad():
    sources = apply_model(model, waveform.unsqueeze(0))

# Save separated sources
num_speakers = sources.shape[1]  # Number of sources
print(f"Detected {num_speakers} speakers.")

for i in range(num_speakers):
    # Extract speaker tensor, move to CPU, and convert to numpy
    speaker = sources[0, i].cpu().numpy()  # shape: [channels, samples]
    # Transpose to shape: [samples, channels]
    speaker = speaker.T
    speaker_path = os.path.join(OUTPUT_DIR, f"speaker_{i+1}.wav")
    sf.write(speaker_path, speaker, sample_rate)  # Save each separated track
    print(f"Saved: {speaker_path}")

print("Separation complete. Separated files are saved in 'Output' folder.")
