"""
Utility functions for dataset manipulation and repetition extraction.
Used for data efficiency experiments with sampled repetitions.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def extract_repetition_units(
    csv_path: str,
    gesture_type: str,
    crop_relax: bool = True,
) -> List[Tuple[int, int]]:
    """
    Extract individual R-O-R or R-C-R units from a CSV file.

    Args:
        csv_path: Path to CSV file with 'gt' column
        gesture_type: 'open' (label=1) or 'close' (label=2)
        crop_relax: If True (default), crop relax to 600 samples (3 sec) on each side.
                    If False, include full natural relax segments using midpoint boundaries
                    between adjacent gestures (matching main_experiment/convergence behavior).

    Returns:
        List of (start_idx, end_idx) tuples for each repetition.
        Each tuple defines a segment that can be extracted from the file.

    Example:
        >>> reps = extract_repetition_units('p15_open_1.csv', 'open')
        >>> print(f"Found {len(reps)} repetitions")
        Found 3 repetitions
        >>> print(f"First rep spans indices {reps[0][0]} to {reps[0][1]}")
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    if 'gt' not in df.columns:
        raise ValueError(f"'gt' column not found in {csv_path}")

    gt = df['gt'].values

    # Determine target label
    if gesture_type == 'open':
        target_label = 1
    elif gesture_type == 'close':
        target_label = 2
    else:
        raise ValueError(f"gesture_type must be 'open' or 'close', got '{gesture_type}'")

    # First pass: find all gesture boundaries
    gestures = []  # list of (gesture_start, gesture_end)
    i = 0
    while i < len(gt) - 1:
        # Find transition from 0 (relax) to target_label (gesture)
        if gt[i] == 0 and gt[i + 1] == target_label:
            gesture_start = i + 1

            # Find transition back to 0 (end of gesture)
            j = gesture_start
            while j < len(gt) and gt[j] == target_label:
                j += 1
            gesture_end = j  # First index back to relax (0)

            gestures.append((gesture_start, gesture_end))

            # Move past this gesture
            i = gesture_end
        else:
            i += 1

    # Second pass: create segments with appropriate relax boundaries
    repetitions = []
    for idx, (gesture_start, gesture_end) in enumerate(gestures):
        if crop_relax:
            # Original behavior: crop to 600 samples (3 sec at 200Hz) on each side
            padding_samples = 600
            start_idx = max(0, gesture_start - padding_samples)
            end_idx = min(len(gt), gesture_end + padding_samples)
        else:
            # New behavior: include full natural relax segments
            # Use midpoint boundaries between adjacent gestures to avoid overlap
            if idx == 0:
                # First gesture: start from beginning of file
                start_idx = 0
            else:
                # Start from midpoint between previous gesture end and this gesture start
                prev_gesture_end = gestures[idx - 1][1]
                start_idx = (prev_gesture_end + gesture_start) // 2

            if idx == len(gestures) - 1:
                # Last gesture: end at end of file
                end_idx = len(gt)
            else:
                # End at midpoint between this gesture end and next gesture start
                next_gesture_start = gestures[idx + 1][0]
                end_idx = (gesture_end + next_gesture_start) // 2

        repetitions.append((start_idx, end_idx))

    return repetitions


def get_paired_repetition_indices(
    participant_folder: str,
    num_sets: int = 4,
    reps_per_set: int = 3,
    crop_relax: bool = True,
) -> Dict[str, Tuple[str, str, List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Generate g_0 through g_11 paired repetition index mapping.

    A paired repetition consists of:
    - One R-O-R unit from an open file
    - One R-C-R unit from a close file (same set, same repetition index)

    Args:
        participant_folder: Path to participant data folder
        num_sets: Number of baseline sets (default 4)
        reps_per_set: Number of repetitions per set (default 3)
        crop_relax: If True (default), crop relax to 600 samples on each side.
                    If False, include full natural relax segments.

    Returns:
        Dict mapping g_i to (open_file, close_file, open_rep_indices, close_rep_indices)
        where i ranges from 0 to 11 (for 4 sets × 3 reps)

    Example:
        >>> indices = get_paired_repetition_indices('~/path/to/p15/')
        >>> g_0 = indices['g_0']
        >>> open_file, close_file, open_reps, close_reps = g_0
        >>> print(f"g_0 uses {open_file} and {close_file}")
    """
    import os

    paired_reps = {}
    g_idx = 0

    for set_num in range(1, num_sets + 1):
        open_file = os.path.join(participant_folder, f"*_open_{set_num}.csv")
        close_file = os.path.join(participant_folder, f"*_close_{set_num}.csv")

        # Find actual files (handle p15/p20 prefix)
        import glob
        open_files = glob.glob(open_file)
        close_files = glob.glob(close_file)

        if len(open_files) != 1 or len(close_files) != 1:
            raise ValueError(
                f"Expected exactly 1 open and 1 close file for set {set_num}, "
                f"found {len(open_files)} open and {len(close_files)} close"
            )

        open_file = open_files[0]
        close_file = close_files[0]

        # Extract repetitions
        open_reps = extract_repetition_units(open_file, 'open', crop_relax=crop_relax)
        close_reps = extract_repetition_units(close_file, 'close', crop_relax=crop_relax)

        if len(open_reps) != reps_per_set or len(close_reps) != reps_per_set:
            raise ValueError(
                f"Expected {reps_per_set} reps per file for set {set_num}, "
                f"found {len(open_reps)} open and {len(close_reps)} close"
            )

        # Create paired repetition indices
        for rep_idx in range(reps_per_set):
            g_name = f"g_{g_idx}"
            paired_reps[g_name] = (
                open_file,
                close_file,
                [open_reps[rep_idx]],  # List with single tuple
                [close_reps[rep_idx]]   # List with single tuple
            )
            g_idx += 1

    return paired_reps


def sample_repetitions(
    paired_reps: Dict,
    budget_k: int,
    num_trials: int = 12,
    seed: int = 42
) -> List[List[str]]:
    """
    Sample repetitions for data efficiency experiment.

    For K=1: Each trial uses a unique g_i (trial i uses g_i)
    For K>1: Each trial randomly samples K repetitions without replacement.
             Same g_i can appear across different trials.

    Args:
        paired_reps: Dict from get_paired_repetition_indices()
        budget_k: Number of paired repetitions per trial (1, 4, or 8)
        num_trials: Number of trials (default 12)
        seed: Random seed for reproducibility

    Returns:
        List of lists, where each inner list contains g_i names for that trial.
        Length of outer list = num_trials
        Length of each inner list = budget_k

    Example:
        >>> paired_reps = get_paired_repetition_indices('~/path/to/p15/')
        >>> k1_samples = sample_repetitions(paired_reps, budget_k=1)
        >>> k4_samples = sample_repetitions(paired_reps, budget_k=4)
    """
    np.random.seed(seed)
    all_g_names = sorted(paired_reps.keys(), key=lambda x: int(x.split('_')[1]))  # g_0 through g_11

    if budget_k > len(all_g_names):
        raise ValueError(
            f"budget_k ({budget_k}) cannot exceed number of available "
            f"paired repetitions ({len(all_g_names)})"
        )

    if budget_k == 1:
        # Each trial uses exactly one unique repetition: trial i uses g_i
        if num_trials > len(all_g_names):
            raise ValueError(
                f"For K=1, num_trials ({num_trials}) cannot exceed "
                f"number of paired repetitions ({len(all_g_names)})"
            )
        return [[all_g_names[i]] for i in range(num_trials)]
    else:
        # Randomly sample K from all 12 for each trial (no replacement within trial)
        trials = []
        for _ in range(num_trials):
            sampled = np.random.choice(
                all_g_names,
                size=budget_k,
                replace=False
            ).tolist()
            trials.append(sampled)
        return trials
