"""
Utility functions for dataset manipulation and repetition extraction.
Used for data efficiency experiments with sampled repetitions.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def extract_repetition_units(csv_path: str, gesture_type: str) -> List[Tuple[int, int]]:
    """
    Extract individual R-O-R or R-C-R units from a CSV file.

    Each unit includes:
    - 3 seconds (600 samples at 200Hz) of relax before gesture
    - Full gesture duration
    - 3 seconds (600 samples at 200Hz) of relax after gesture

    Args:
        csv_path: Path to CSV file with 'gt' column
        gesture_type: 'open' (label=1) or 'close' (label=2)

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

    repetitions = []

    # Find all occurrences of 0 -> target_label -> 0 pattern
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

            # Add 3 seconds (600 samples at 200Hz) before and after
            # Clamp to valid indices
            padding_samples = 600
            start_idx = max(0, gesture_start - padding_samples)
            end_idx = min(len(gt), gesture_end + padding_samples)

            repetitions.append((start_idx, end_idx))

            # Move past this gesture
            i = gesture_end
        else:
            i += 1

    return repetitions


def get_paired_repetition_indices(
    participant_folder: str,
    num_sets: int = 4,
    reps_per_set: int = 3
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
        open_reps = extract_repetition_units(open_file, 'open')
        close_reps = extract_repetition_units(close_file, 'close')

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
    all_g_names = sorted(paired_reps.keys())  # g_0 through g_11

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
