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

