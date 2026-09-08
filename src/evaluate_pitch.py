import numpy as np

CACHE_PATH = "data/processed/carnatic/song1_pitch_cache.npz"


def load_data():
    data = np.load(CACHE_PATH)

    return (
        data["pyin_times"],
        data["pyin_f0"],
        data["pyin_confidence"],
        data["crepe_times"],
        data["crepe_f0"],
        data["crepe_confidence"]
    )


def evaluate_tracker(name, f0, confidence, times):
    valid = ~np.isnan(f0)

    valid_f0 = f0[valid]
    valid_confidence = confidence[valid]

    if len(valid_f0) == 0:
        print(f"\n{name}: No valid pitch frames")
        return

    # Pitch changes between consecutive valid frames
    pitch_changes = np.abs(
        np.diff(valid_f0)
    )

    # Ignore tiny numerical changes and count large jumps
    large_jumps = pitch_changes > 50

    print(f"\n{'=' * 45}")
    print(f"{name}")
    print(f"{'=' * 45}")

    print(f"Total frames       : {len(f0)}")
    print(f"Valid pitch frames : {len(valid_f0)}")

    print(
        f"Voiced coverage    : "
        f"{100 * len(valid_f0) / len(f0):.2f}%"
    )

    print(
        f"Mean confidence    : "
        f"{np.mean(valid_confidence):.3f}"
    )

    print(
        f"Median confidence  : "
        f"{np.median(valid_confidence):.3f}"
    )

    print(
        f"Median pitch       : "
        f"{np.median(valid_f0):.2f} Hz"
    )

    print(
        f"Pitch range        : "
        f"{np.min(valid_f0):.2f} - "
        f"{np.max(valid_f0):.2f} Hz"
    )

    print(
        f"Large jumps >50Hz  : "
        f"{np.sum(large_jumps)}"
    )

    if len(pitch_changes) > 0:
        print(
            f"Median frame jump  : "
            f"{np.median(pitch_changes):.2f} Hz"
        )


def compare_trackers(
    pyin_times,
    pyin_f0,
    crepe_times,
    crepe_f0
):
    # Find timestamps common to both trackers
    common_start = max(
        pyin_times[0],
        crepe_times[0]
    )

    common_end = min(
        pyin_times[-1],
        crepe_times[-1]
    )

    pyin_mask = (
        (pyin_times >= common_start) &
        (pyin_times <= common_end) &
        ~np.isnan(pyin_f0)
    )

    crepe_mask = (
        (crepe_times >= common_start) &
        (crepe_times <= common_end) &
        ~np.isnan(crepe_f0)
    )

    pyin_valid_times = pyin_times[pyin_mask]
    pyin_valid_f0 = pyin_f0[pyin_mask]

    crepe_valid_times = crepe_times[crepe_mask]
    crepe_valid_f0 = crepe_f0[crepe_mask]

    # Interpolate CREPE onto pYIN timestamps
    if len(crepe_valid_times) > 1:
        crepe_interpolated = np.interp(
            pyin_valid_times,
            crepe_valid_times,
            crepe_valid_f0
        )

        differences = np.abs(
            pyin_valid_f0 - crepe_interpolated
        )

        print(f"\n{'=' * 45}")
        print("pYIN vs CREPE")
        print(f"{'=' * 45}")

        print(
            f"Median absolute difference : "
            f"{np.median(differences):.2f} Hz"
        )

        print(
            f"Mean absolute difference   : "
            f"{np.mean(differences):.2f} Hz"
        )

        print(
            f"Frames differing >50 Hz   : "
            f"{np.sum(differences > 50)}"
        )

        print(
            f"Frames differing >100 Hz  : "
            f"{np.sum(differences > 100)}"
        )


if __name__ == "__main__":

    print("Loading cached pitch data...")

    (
        pyin_times,
        pyin_f0,
        pyin_confidence,
        crepe_times,
        crepe_f0,
        crepe_confidence
    ) = load_data()

    evaluate_tracker(
        "pYIN",
        pyin_f0,
        pyin_confidence,
        pyin_times
    )

    evaluate_tracker(
        "CREPE",
        crepe_f0,
        crepe_confidence,
        crepe_times
    )

    compare_trackers(
        pyin_times,
        pyin_f0,
        crepe_times,
        crepe_f0
    )