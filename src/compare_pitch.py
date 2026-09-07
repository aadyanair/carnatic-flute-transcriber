import librosa
import crepe
import numpy as np
import matplotlib.pyplot as plt


AUDIO_PATH = "data/processed/carnatic/song1.wav"
TARGET_SR = 16000


def extract_pyin(audio, sr):
    print("Running pYIN...")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=2048,
        hop_length=160
    )

    times = librosa.times_like(
        f0,
        sr=sr,
        hop_length=160
    )

    return times, f0


def extract_crepe(audio, sr):
    print("Running CREPE...")

    time, frequency, confidence, activation = crepe.predict(
        audio,
        sr,
        viterbi=True,
        step_size=10
    )

    return time, frequency, confidence


def plot_comparison(
    pyin_times,
    pyin_f0,
    crepe_times,
    crepe_f0
):
    plt.figure(figsize=(15, 6))

    plt.plot(
        pyin_times,
        pyin_f0,
        label="pYIN",
        alpha=0.7
    )

    plt.plot(
        crepe_times,
        crepe_f0,
        label="CREPE",
        alpha=0.7
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("pYIN vs CREPE — Flute Pitch Tracking")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print(f"Loading: {AUDIO_PATH}")

    audio, sr = librosa.load(
        AUDIO_PATH,
        sr=TARGET_SR,
        mono=True
    )

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")

    pyin_times, pyin_f0 = extract_pyin(
        audio,
        sr
    )

    crepe_times, crepe_f0, crepe_confidence = extract_crepe(
        audio,
        sr
    )

    plot_comparison(
        pyin_times,
        pyin_f0,
        crepe_times,
        crepe_f0
    )