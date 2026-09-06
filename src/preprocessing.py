import librosa
import soundfile as sf
from pathlib import Path


TARGET_SR = 16000

SUPPORTED_FORMATS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg"
}


def preprocess_audio(input_path, output_path):
    print(f"\nProcessing: {input_path}")

    # Load audio
    audio, sr = librosa.load(
        input_path,
        sr=TARGET_SR,
        mono=True
    )

    original_duration = len(audio) / sr

    # Trim leading and trailing silence
    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    processed_duration = len(audio_trimmed) / sr

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed audio
    sf.write(
        output_path,
        audio_trimmed,
        sr
    )

    print(f"  Sample rate : {sr} Hz")
    print(f"  Original    : {original_duration:.2f} sec")
    print(f"  Processed   : {processed_duration:.2f} sec")
    print(f"  Removed     : {original_duration - processed_duration:.2f} sec")
    print(f"  Saved to    : {output_path}")


def process_directory(input_directory, output_directory):
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    if not input_directory.exists():
        print(f"Directory not found: {input_directory}")
        return

    audio_files = [
        file for file in input_directory.iterdir()
        if file.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not audio_files:
        print(f"No audio files found in {input_directory}")
        return

    print(f"\nFound {len(audio_files)} audio file(s) in {input_directory}")

    for input_file in audio_files:

        output_file = output_directory / f"{input_file.stem}.wav"

        preprocess_audio(
            input_file,
            output_file
        )


if __name__ == "__main__":

    process_directory(
        "data/raw/carnatic",
        "data/processed/carnatic"
    )

    process_directory(
        "data/raw/other",
        "data/processed/other"
    )