# 🚧 Project Overview: RealTimeVoice
_Generated on 2025-05-13T18:49:46.738843_

## 📂 Folder Structure
```
├── Models/
│   ├── student_embeddings.pkl
│   ├── student_embeddings10.pkl
│   ├── student_embeddings3.pkl
│   ├── student_embeddings5.pkl
│   ├── student_embeddings6.pkl
│   └── student_embeddings7.pkl
├── OPUStoWAV.py
├── Output/
│   ├── speaker_1.wav
│   ├── speaker_2.wav
│   ├── speaker_3.wav
│   └── speaker_4.wav
├── RealTimeVoice.py
├── RealTimeVoice.pyproj
├── RealTimeVoice.sln
└── SourceSeparation.py
```

## 📄 Code Files

### `OPUStoWAV.py`
- **Lines:** 34
- **Last Modified:** 2025-04-06T22:08:19.068680

```
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pydub import AudioSegment

def convert_opus_to_wav():
    file_path = filedialog.askopenfilename(
        filetypes=[("Opus Audio Files", "*.opus")]
    )
    if not file_path:
        return

    try:
        audio = AudioSegment.from_file(file_path, format="opus")
        output_path = os.path.splitext(file_path)[0] + ".wav"
        audio.export(output_path, format="wav")
        messagebox.showinfo("Success", f"Converted to:\n{output_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("OPUS to WAV Converter")
root.geometry("300x150")
root.resizable(False, False)

label = tk.Label(root, text="Click the button below to select a .opus file", wraplength=250)
label.pack(pady=20)

btn = tk.Button(root, text="Convert OPUS to WAV", command=convert_opus_to_wav)
btn.pack(pady=10)

root.mainloop()


```

### `RealTimeVoice.py`
- **Lines:** 216
- **Last Modified:** 2025-04-07T10:34:42.590540

```
﻿import os
import re
import numpy as np
import torch
import librosa
import pickle
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from huggingface_hub import hf_hub_download

# ------------------------- Configuration Variables -------------------------
SR = 16000                           # Sampling rate (ECAPA2 is trained at 16 kHz)
MODEL_PATH = os.path.join("Models", "student_embeddings.pkl") # Path to the saved embeddings pickle file
INDIVIDUAL_MARGIN = 0.5              # Margin above the dynamic threshold to choose an individual candidate
# ------------------------------------------------------------------------------

# Download and load the ECAPA2 model from Hugging Face
model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecapa2 = torch.jit.load(model_file, map_location=torch.device("cpu"))
ecapa2 = ecapa2.to(device)
print("ECAPA2 model loaded on:", device)

def extract_embeddings(audio, sr=SR):
    """Extract embeddings from a numpy array of audio."""
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    embedding = ecapa2(audio_tensor).detach().cpu().numpy()
    return embedding

def split_composite_name(composite):
    """
    Split a composite string into individual speaker names.
    Assumes each name starts with a capital letter.
    Splits on spaces and boundaries where a lowercase letter is followed by an uppercase letter.
    """
    parts = composite.split(" ")
    segments = []
    for part in parts:
        # The first regex splits where a lowercase letter is followed by an uppercase letter.
        subparts = re.split(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', part)
        segments.extend(subparts)
    merged = []
    i = 0
    while i < len(segments):
        current = segments[i]
        if len(current) == 1:
            combo = current
            j = i + 1
            while j < len(segments) and len(segments[j]) == 1:
                combo += " " + segments[j]
                j += 1
            merged.append(combo)
            i = j
        else:
            merged.append(current)
            i += 1
    return merged

def format_speaker_list(speakers):
    """Format a list of speakers into a natural language string."""
    if not speakers:
        return ""
    if len(speakers) == 1:
        return speakers[0]
    if len(speakers) == 2:
        return speakers[0] + " and " + speakers[1]
    return ", ".join(speakers[:-1]) + " and " + speakers[-1]

def get_pair_similarity(s1, s2, similarities):
    """
    Look up the similarity score for a pair of speakers.
    Checks both orders (e.g. "ArunJuan" and "JuanArun").
    """
    key1 = s1 + s2
    key2 = s2 + s1
    if key1 in similarities:
        return similarities[key1]
    elif key2 in similarities:
        return similarities[key2]
    else:
        return 0

def average_pairwise(speakers, similarities):
    """Compute the average pairwise similarity for a list of speakers."""
    pairs = []
    for i in range(len(speakers)):
        for j in range(i+1, len(speakers)):
            pairs.append(get_pair_similarity(speakers[i], speakers[j], similarities))
    return np.mean(pairs) if pairs else 0

def select_composite_candidate(similarities, ind_threshold, pairwise_threshold):
    """
    Among all candidate keys in the similarities dictionary, select the composite candidate 
    (with >1 speaker) that:
      - Has every individual's similarity >= ind_threshold,
      - Has an average pairwise similarity >= pairwise_threshold,
      - And is selected by prioritizing the number of speakers first, then the average individual score.
    If no composite candidate qualifies, return the individual candidate with the highest score.
    """
    valid_candidates = []
    for candidate, _ in similarities.items():
        speakers = split_composite_name(candidate)
        if len(speakers) > 1:
            if not all(similarities.get(sp, 0) >= ind_threshold for sp in speakers):
                continue
            avg_pair_sim = average_pairwise(speakers, similarities)
            if avg_pair_sim < pairwise_threshold:
                continue
            avg_individual = np.mean([similarities.get(sp, 0) for sp in speakers])
            valid_candidates.append((candidate, avg_individual, len(speakers)))
    if valid_candidates:
        # Prioritize by count of speakers, then by average individual score.
        best_candidate, best_avg, best_count = max(valid_candidates, key=lambda x: (x[2], x[1]))
        return best_candidate, best_avg
    else:
        best_candidate = max(similarities, key=similarities.get)
        return best_candidate, similarities[best_candidate]

def dynamic_thresholds(similarities):
    """
    Compute dynamic thresholds based on the distribution of confidence scores.
    We calculate the minimum and maximum confidence scores and then choose a threshold
    that is a fixed fraction above the minimum.
    Adjust the factor (0.3 in this example) as needed.
    """
    scores = np.array(list(similarities.values()))
    min_conf = scores.min()
    max_conf = scores.max()
    dynamic_threshold = min_conf + 0.32 * (max_conf - min_conf)
    return dynamic_threshold, dynamic_threshold  # Returning the same for both individual and pairwise

def compare_embeddings(embedding, student_embeddings):
    """
    Compute cosine similarities between the input embedding and stored embeddings.
    Then select the best candidate by first checking if an individual speaker stands out.
    If not, use composite candidate selection based on dynamic thresholds.
    Finally, split the chosen candidate into individual speaker names.
    """
    similarities = {
        student: cosine_similarity(embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
        for student, emb in student_embeddings.items()
    }
    for student, conf in similarities.items():
        print(f"Confidence score for {student}: {conf:.2f}")
    
    # Calculate dynamic thresholds based on the current file's score distribution.
    dynamic_ind_threshold, dynamic_pair_threshold = dynamic_thresholds(similarities)
    print(f"Dynamic Individual Threshold: {dynamic_ind_threshold:.2f}")
    print(f"Dynamic Pairwise Threshold: {dynamic_pair_threshold:.2f}")
    
    # Check if one individual speaker clearly stands out
    max_candidate, max_conf = max(similarities.items(), key=lambda x: x[1])
    if max_conf >= dynamic_ind_threshold + INDIVIDUAL_MARGIN:
        # Even if the key is composite (like 'GeorgeJuan'), split it into individual speakers.
        split_candidate = split_composite_name(max_candidate)
        return split_candidate, max_conf

    # Otherwise, proceed with composite candidate selection.
    composite, composite_conf = select_composite_candidate(similarities, dynamic_ind_threshold, dynamic_pair_threshold)
    speakers = split_composite_name(composite)
    return speakers, composite_conf

def process_audio_file(file_path, student_embeddings):
    """
    Process the audio file:
      - Convert to WAV if necessary,
      - Load and normalize the audio,
      - Extract embeddings,
      - Compare against stored embeddings.
    Returns the identified speaker list and a confidence score.
    """
    if not file_path.lower().endswith(".wav"):
        audio = AudioSegment.from_file(file_path)
        file_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio.export(file_path, format="wav")
    audio, sr = librosa.load(file_path, sr=SR)
    audio = audio / np.max(np.abs(audio))
    audio = audio.astype(np.float32)
    embeddings = extract_embeddings(audio, sr)
    speakers, confidence = compare_embeddings(embeddings, student_embeddings)
    return speakers, confidence

def load_saved_model():
    """Load the saved student embeddings from a pickle file."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            student_embeddings = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
        return student_embeddings
    else:
        print(f"Saved model not found at {MODEL_PATH}.")
        return None

def main():
    student_embeddings = load_saved_model()
    if student_embeddings is None:
        print("No saved model found. Please train your model and save the embeddings first.")
        return

    root = tk.Tk()
    root.withdraw()
    file_path = tk.filedialog.askopenfilename(
        title="Select Recorded Audio File (WAV format)",
        filetypes=[("WAV files", "*.wav")]
    )
    if not file_path:
        print("No file selected.")
        return
    speakers, confidence = process_audio_file(file_path, student_embeddings)
    formatted = format_speaker_list(speakers)
    print(f"Identified Speakers: {formatted} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()

```

### `RealTimeVoice.sln`
- **Lines:** 23
- **Last Modified:** 2025-03-27T19:18:46.013144

```
﻿
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.13.35818.85 d17.13
MinimumVisualStudioVersion = 10.0.40219.1
Project("{888888A0-9F3D-457C-B088-3A5042F75D52}") = "RealTimeVoice", "RealTimeVoice.pyproj", "{EFE3B2F0-F7FF-4525-8AA8-18B3A5156502}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|Any CPU = Debug|Any CPU
		Release|Any CPU = Release|Any CPU
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{EFE3B2F0-F7FF-4525-8AA8-18B3A5156502}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
		{EFE3B2F0-F7FF-4525-8AA8-18B3A5156502}.Release|Any CPU.ActiveCfg = Release|Any CPU
	EndGlobalSection
	GlobalSection(SolutionProperties) = preSolution
		HideSolutionNode = FALSE
	EndGlobalSection
	GlobalSection(ExtensibilityGlobals) = postSolution
		SolutionGuid = {1426C3EC-A278-4C12-BC7C-4ADEDF276ADE}
	EndGlobalSection
EndGlobal

```

### `SourceSeparation.py`
- **Lines:** 53
- **Last Modified:** 2025-04-03T22:01:24.616241

```
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

```
