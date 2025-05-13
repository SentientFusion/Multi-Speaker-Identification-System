import os
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
