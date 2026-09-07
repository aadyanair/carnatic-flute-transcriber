import librosa
import numpy as np
import matplotlib.pyplot as plt


AUDIO_PATH = "data/processed/carnatic/song1.wav"

def extract_pitch(audio_path):
    print(f"Loading: {audio_path}")

    audio, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio) /sr:.2f} seconds")

    # Extract fundamental frequency using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=2048,
        hop_length=160
    )

    # Time corresponding to each F0 value
    times = librosa.times_like(
        f0,
        sr=sr,
        hop_length=160
    )

    return times, f0, voiced_flag, voiced_probs

def plot_pitch(times, f0):
    plt.figure(figsize=(14, 5))

    plt.plot(
        times,
        f0
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Flute pitch contous - pYIN")

    plt.grid()

    plt.show()

if __name__ == "__main__":

    times, f0, voiced_flag, voiced_probs = extract_pitch(AUDIO_PATH)

    plot_pitch(times, f0)