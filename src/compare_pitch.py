import librosa
import crepe
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path

AUDIO_PATH = "data/processed/carnatic/song1.wav"
CACHE_PATH = "data/processed/carnatic/song1_pitch_cache.npz"

TARGET_SR = 16000
CONFIDENCE_THRESHOLD = 0.5


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

    f0_filtered = f0.copy()
    f0_filtered[voiced_prob < CONFIDENCE_THRESHOLD] = np.nan

    return times, f0_filtered, voiced_prob


def extract_crepe(audio, sr):
    print("Running CREPE...")

    time, frequency, confidence, activation = crepe.predict(
        audio,
        sr,
        viterbi=True,
        step_size=10,
        verbose=0
    )

    frequency_filtered = frequency.copy()
    frequency_filtered[confidence < CONFIDENCE_THRESHOLD] = np.nan

    return time, frequency_filtered, confidence


def save_cache(
    pyin_times,
    pyin_f0,
    pyin_confidence,
    crepe_times,
    crepe_f0,
    crepe_confidence
):
    Path(CACHE_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        CACHE_PATH,
        pyin_times=pyin_times,
        pyin_f0=pyin_f0,
        pyin_confidence=pyin_confidence,
        crepe_times=crepe_times,
        crepe_f0=crepe_f0,
        crepe_confidence=crepe_confidence
    )

    print(f"\nPitch results saved to:")
    print(CACHE_PATH)


def load_cache():
    print(f"\nLoading cached pitch results:")
    print(CACHE_PATH)

    data = np.load(CACHE_PATH)

    return (
        data["pyin_times"],
        data["pyin_f0"],
        data["pyin_confidence"],
        data["crepe_times"],
        data["crepe_f0"],
        data["crepe_confidence"]
    )


def plot_comparison(
    pyin_times,
    pyin_f0,
    crepe_times,
    crepe_f0,
    start_time=10,
    end_time=20
):
    plt.figure(figsize=(15, 6))

    pyin_mask = (
        (pyin_times >= start_time) &
        (pyin_times <= end_time)
    )

    crepe_mask = (
        (crepe_times >= start_time) &
        (crepe_times <= end_time)
    )

    plt.plot(
        pyin_times[pyin_mask],
        pyin_f0[pyin_mask],
        label="pYIN"
    )

    plt.plot(
        crepe_times[crepe_mask],
        crepe_f0[crepe_mask],
        label="CREPE"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.title(
        f"pYIN vs CREPE — "
        f"{start_time}s to {end_time}s"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_spectrogram_with_pitch(
    audio,
    sr,
    pyin_times,
    pyin_f0,
    crepe_times,
    crepe_f0,
    start_time=10,
    end_time=20
):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    audio_segment = audio[start_sample:end_sample]

    plt.figure(figsize=(15, 7))

    D = librosa.stft(
        audio_segment,
        n_fft=2048,
        hop_length=160
    )

    D_db = librosa.amplitude_to_db(
        np.abs(D),
        ref=np.max
    )

    librosa.display.specshow(
        D_db,
        sr=sr,
        hop_length=160,
        x_axis="time",
        y_axis="hz"
    )

    pyin_mask = (
        (pyin_times >= start_time) &
        (pyin_times <= end_time)
    )

    crepe_mask = (
        (crepe_times >= start_time) &
        (crepe_times <= end_time)
    )

    plt.plot(
        pyin_times[pyin_mask] - start_time,
        pyin_f0[pyin_mask],
        label="pYIN"
    )

    plt.plot(
        crepe_times[crepe_mask] - start_time,
        crepe_f0[crepe_mask],
        label="CREPE"
    )

    plt.ylim(100, 1000)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.title(
        f"Pitch Tracking Over Spectrogram "
        f"({start_time}s–{end_time}s)"
    )

    plt.legend()
    plt.colorbar(format="%+2.0f dB")

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

    # --------------------------------------------------
    # LOAD CACHE IF IT EXISTS
    # --------------------------------------------------

    if Path(CACHE_PATH).exists():

        print("\nCached pitch data found.")
        print("Skipping pYIN and CREPE extraction.")

        (
            pyin_times,
            pyin_f0,
            pyin_confidence,
            crepe_times,
            crepe_f0,
            crepe_confidence
        ) = load_cache()

    # --------------------------------------------------
    # OTHERWISE RUN PITCH EXTRACTION
    # --------------------------------------------------

    else:

        pyin_times, pyin_f0, pyin_confidence = extract_pyin(
            audio,
            sr
        )

        crepe_times, crepe_f0, crepe_confidence = extract_crepe(
            audio,
            sr
        )

        save_cache(
            pyin_times,
            pyin_f0,
            pyin_confidence,
            crepe_times,
            crepe_f0,
            crepe_confidence
        )

    # --------------------------------------------------
    # PLOT RESULTS
    # --------------------------------------------------

    plot_comparison(
        pyin_times,
        pyin_f0,
        crepe_times,
        crepe_f0,
        start_time=10,
        end_time=20
    )

    plot_spectrogram_with_pitch(
        audio,
        sr,
        pyin_times,
        pyin_f0,
        crepe_times,
        crepe_f0,
        start_time=10,
        end_time=20
    )