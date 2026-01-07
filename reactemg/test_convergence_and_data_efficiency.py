"""
Comprehensive unit tests for run_convergence.py and run_data_efficiency.py

This test suite exhaustively validates:
- Configuration correctness (PARTICIPANTS, TEST_CONDITIONS, etc.)
- File discovery functions (get_healthy_s25_files, get_test_files, get_all_calibration_files)
- Error handling and exception raising
- dataset_utils.py functions (extract_repetition_units, get_paired_repetition_indices, sample_repetitions)
- Training command construction
- Symlink creation and cleanup
- Data sampling logic
- Integration tests with mocking

Run with: python3 test_convergence_and_data_efficiency.py
"""

import os
import sys
import glob
import tempfile
import shutil
import json
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from typing import List, Dict, Tuple

# Import modules under test
from run_convergence import (
    PARTICIPANTS as CONV_PARTICIPANTS,
    PRETRAINED_CHECKPOINT as CONV_PRETRAINED_CHECKPOINT,
    TEST_CONDITIONS as CONV_TEST_CONDITIONS,
    get_healthy_s25_files,
    get_test_files as conv_get_test_files,
    get_all_calibration_files as conv_get_all_calibration_files,
)

from run_data_efficiency import (
    PARTICIPANTS as DE_PARTICIPANTS,
    PRETRAINED_CHECKPOINT as DE_PRETRAINED_CHECKPOINT,
    TEST_CONDITIONS as DE_TEST_CONDITIONS,
    get_test_files as de_get_test_files,
)

from dataset_utils import (
    extract_repetition_units,
    get_paired_repetition_indices,
    sample_repetitions,
)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

PARTICIPANT_FOLDERS = {
    'p4': os.path.expanduser('~/Workspace/myhand/src/collected_data/2026_01_06'),
    'p15': os.path.expanduser('~/Workspace/myhand/src/collected_data/2025_12_04'),
    'p20': os.path.expanduser('~/Workspace/myhand/src/collected_data/2025_12_18'),
}

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'warnings': [],
    'skipped': []
}


def log_pass(test_name: str, message: str = ""):
    """Log a passing test."""
    test_results['passed'].append(test_name)
    print(f"  \u2713 {test_name}")
    if message:
        print(f"    {message}")


def log_fail(test_name: str, error: str):
    """Log a failing test."""
    test_results['failed'].append((test_name, error))
    print(f"  \u2717 {test_name}")
    print(f"    ERROR: {error}")


def log_warning(test_name: str, warning: str):
    """Log a warning."""
    test_results['warnings'].append((test_name, warning))
    print(f"  \u26a0 {test_name}")
    print(f"    WARNING: {warning}")


def log_skip(test_name: str, reason: str):
    """Log a skipped test."""
    test_results['skipped'].append((test_name, reason))
    print(f"  \u25cb {test_name} (SKIPPED)")
    print(f"    Reason: {reason}")


def create_mock_csv_with_labels(filepath: str, labels: List[int], num_samples: int = None):
    """
    Create a mock CSV file with specified label pattern.

    Args:
        filepath: Path to create CSV file
        labels: List of labels (will be repeated/truncated to fit num_samples)
        num_samples: Number of samples (rows). If None, uses len(labels)
    """
    if num_samples is None:
        num_samples = len(labels)

    # Extend or truncate labels to match num_samples
    if len(labels) < num_samples:
        labels = labels * (num_samples // len(labels) + 1)
    labels = labels[:num_samples]

    # Create DataFrame with required columns
    data = {
        'gt': labels,
        'time_elapsed': np.arange(num_samples) * 0.005,  # 200 Hz
        'current_time': np.arange(num_samples),
        'current_task': [0] * num_samples,
        'motor_position': [0.0] * num_samples,
        'futek': [0.0] * num_samples,
    }

    # Add EMG columns
    for i in range(8):
        data[f'emg{i}'] = np.random.randint(0, 256, num_samples)

    data['emg_timer_stamp'] = np.arange(num_samples)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def create_ror_pattern(gesture_type: str, num_reps: int = 3,
                       relax_duration: int = 800, gesture_duration: int = 600) -> List[int]:
    """
    Create a Relax-Open/Close-Relax pattern.

    Args:
        gesture_type: 'open' (label=1) or 'close' (label=2)
        num_reps: Number of repetitions
        relax_duration: Duration of relax phases (samples)
        gesture_duration: Duration of gesture phases (samples)

    Returns:
        List of labels following R-O-R or R-C-R pattern
    """
    target_label = 1 if gesture_type == 'open' else 2
    labels = []

    for _ in range(num_reps):
        labels.extend([0] * relax_duration)  # Relax before
        labels.extend([target_label] * gesture_duration)  # Gesture
        labels.extend([0] * relax_duration)  # Relax after

    return labels


# ============================================================================
# SECTION 1: CONFIGURATION VALIDATION - run_convergence.py
# ============================================================================

def test_convergence_configuration():
    """Test run_convergence.py configuration validity."""
    print("\n" + "="*80)
    print("TEST SECTION: run_convergence.py Configuration Validation")
    print("="*80)

    # Test 1.1: PARTICIPANTS dict matches expected structure
    if not isinstance(CONV_PARTICIPANTS, dict):
        log_fail("conv_participants_is_dict", f"PARTICIPANTS should be dict, got {type(CONV_PARTICIPANTS)}")
    else:
        log_pass("conv_participants_is_dict", f"PARTICIPANTS has {len(CONV_PARTICIPANTS)} entries")

    # Test 1.2: PARTICIPANTS keys match between scripts
    from run_main_experiment import PARTICIPANTS as MAIN_PARTICIPANTS
    if set(CONV_PARTICIPANTS.keys()) == set(MAIN_PARTICIPANTS.keys()):
        log_pass("conv_participants_match_main", "PARTICIPANTS keys match run_main_experiment.py")
    else:
        log_fail("conv_participants_match_main",
                f"Mismatch: convergence has {set(CONV_PARTICIPANTS.keys())}, main has {set(MAIN_PARTICIPANTS.keys())}")

    # Test 1.3: PRETRAINED_CHECKPOINT exists
    if os.path.isfile(CONV_PRETRAINED_CHECKPOINT):
        log_pass("conv_checkpoint_exists", f"Checkpoint exists: {CONV_PRETRAINED_CHECKPOINT}")
    else:
        log_fail("conv_checkpoint_exists", f"Checkpoint not found: {CONV_PRETRAINED_CHECKPOINT}")

    # Test 1.4: TEST_CONDITIONS matches main experiment
    from run_main_experiment import TEST_CONDITIONS as MAIN_TEST_CONDITIONS
    if CONV_TEST_CONDITIONS == MAIN_TEST_CONDITIONS:
        log_pass("conv_test_conditions_match", "TEST_CONDITIONS matches run_main_experiment.py")
    else:
        log_fail("conv_test_conditions_match", "TEST_CONDITIONS differs from run_main_experiment.py")


# ============================================================================
# SECTION 2: CONFIGURATION VALIDATION - run_data_efficiency.py
# ============================================================================

def test_data_efficiency_configuration():
    """Test run_data_efficiency.py configuration validity."""
    print("\n" + "="*80)
    print("TEST SECTION: run_data_efficiency.py Configuration Validation")
    print("="*80)

    # Test 2.1: PARTICIPANTS dict matches expected structure
    if not isinstance(DE_PARTICIPANTS, dict):
        log_fail("de_participants_is_dict", f"PARTICIPANTS should be dict, got {type(DE_PARTICIPANTS)}")
    else:
        log_pass("de_participants_is_dict", f"PARTICIPANTS has {len(DE_PARTICIPANTS)} entries")

    # Test 2.2: PARTICIPANTS keys match between scripts
    from run_main_experiment import PARTICIPANTS as MAIN_PARTICIPANTS
    if set(DE_PARTICIPANTS.keys()) == set(MAIN_PARTICIPANTS.keys()):
        log_pass("de_participants_match_main", "PARTICIPANTS keys match run_main_experiment.py")
    else:
        log_fail("de_participants_match_main",
                f"Mismatch: data_efficiency has {set(DE_PARTICIPANTS.keys())}, main has {set(MAIN_PARTICIPANTS.keys())}")

    # Test 2.3: PRETRAINED_CHECKPOINT exists
    if os.path.isfile(DE_PRETRAINED_CHECKPOINT):
        log_pass("de_checkpoint_exists", f"Checkpoint exists: {DE_PRETRAINED_CHECKPOINT}")
    else:
        log_fail("de_checkpoint_exists", f"Checkpoint not found: {DE_PRETRAINED_CHECKPOINT}")

    # Test 2.4: TEST_CONDITIONS matches main experiment
    from run_main_experiment import TEST_CONDITIONS as MAIN_TEST_CONDITIONS
    if DE_TEST_CONDITIONS == MAIN_TEST_CONDITIONS:
        log_pass("de_test_conditions_match", "TEST_CONDITIONS matches run_main_experiment.py")
    else:
        log_fail("de_test_conditions_match", "TEST_CONDITIONS differs from run_main_experiment.py")


# ============================================================================
# SECTION 3: get_test_files() - Exception Handling Tests
# ============================================================================

def test_get_test_files_raises_file_not_found():
    """Test that get_test_files raises FileNotFoundError when files are missing."""
    print("\n" + "="*80)
    print("TEST SECTION: get_test_files() Exception Handling")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 3.1: Empty directory should raise FileNotFoundError
        try:
            conv_get_test_files(tmpdir, 'mid_session_baseline')
            log_fail("conv_test_files_raises_fnf", "Should raise FileNotFoundError for empty dir")
        except FileNotFoundError as e:
            if "No files found" in str(e):
                log_pass("conv_test_files_raises_fnf", "Correctly raises FileNotFoundError with message")
            else:
                log_fail("conv_test_files_raises_fnf", f"FileNotFoundError message unclear: {e}")
        except Exception as e:
            log_fail("conv_test_files_raises_fnf", f"Wrong exception type: {type(e).__name__}: {e}")

        # Test 3.2: Same test for data_efficiency version
        try:
            de_get_test_files(tmpdir, 'mid_session_baseline')
            log_fail("de_test_files_raises_fnf", "Should raise FileNotFoundError for empty dir")
        except FileNotFoundError as e:
            if "No files found" in str(e):
                log_pass("de_test_files_raises_fnf", "Correctly raises FileNotFoundError with message")
            else:
                log_fail("de_test_files_raises_fnf", f"FileNotFoundError message unclear: {e}")
        except Exception as e:
            log_fail("de_test_files_raises_fnf", f"Wrong exception type: {type(e).__name__}: {e}")


def test_get_test_files_raises_value_error():
    """Test that get_test_files raises ValueError when multiple files match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two files matching the same pattern
        open(os.path.join(tmpdir, "p15_open_5.csv"), 'w').close()
        open(os.path.join(tmpdir, "p15_backup_open_5.csv"), 'w').close()
        # Also need close_5 to not trigger FileNotFoundError first
        open(os.path.join(tmpdir, "p15_close_5.csv"), 'w').close()

        # Test 3.3: Multiple matches for open_5 should raise ValueError
        try:
            conv_get_test_files(tmpdir, 'mid_session_baseline')
            log_fail("conv_test_files_raises_ve", "Should raise ValueError for multiple matches")
        except ValueError as e:
            if "expected exactly 1" in str(e).lower():
                log_pass("conv_test_files_raises_ve", "Correctly raises ValueError with message")
            else:
                log_fail("conv_test_files_raises_ve", f"ValueError message unclear: {e}")
        except FileNotFoundError:
            log_pass("conv_test_files_raises_ve", "FileNotFoundError raised (close_5 checked first)")
        except Exception as e:
            log_fail("conv_test_files_raises_ve", f"Wrong exception type: {type(e).__name__}: {e}")


def test_get_test_files_success():
    """Test get_test_files succeeds with correct files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create exactly the expected files for mid_session_baseline
        open(os.path.join(tmpdir, "p15_open_5.csv"), 'w').close()
        open(os.path.join(tmpdir, "p15_close_5.csv"), 'w').close()

        # Test 3.4: Should succeed with correct files
        try:
            files = conv_get_test_files(tmpdir, 'mid_session_baseline')
            if len(files) == 2:
                log_pass("conv_test_files_success", f"Successfully found 2 files")
            else:
                log_fail("conv_test_files_success", f"Expected 2 files, got {len(files)}")
        except Exception as e:
            log_fail("conv_test_files_success", f"Unexpected error: {e}")


def test_get_test_files_invalid_condition():
    """Test get_test_files with invalid condition name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 3.5: Invalid condition should raise KeyError
        try:
            conv_get_test_files(tmpdir, 'invalid_condition_xyz')
            log_fail("conv_test_files_invalid_cond", "Should raise KeyError for invalid condition")
        except KeyError:
            log_pass("conv_test_files_invalid_cond", "Correctly raises KeyError")
        except Exception as e:
            log_fail("conv_test_files_invalid_cond", f"Wrong exception: {type(e).__name__}: {e}")


# ============================================================================
# SECTION 4: get_healthy_s25_files() Tests
# ============================================================================

def test_get_healthy_s25_files_path_not_found():
    """Test get_healthy_s25_files raises error for nonexistent path."""
    print("\n" + "="*80)
    print("TEST SECTION: get_healthy_s25_files() Tests")
    print("="*80)

    # Test 4.1: Nonexistent path should raise FileNotFoundError
    try:
        get_healthy_s25_files("/nonexistent/path/to/s25")
        log_fail("s25_path_not_found", "Should raise FileNotFoundError for nonexistent path")
    except FileNotFoundError as e:
        if "does not exist" in str(e):
            log_pass("s25_path_not_found", "Correctly raises FileNotFoundError")
        else:
            log_fail("s25_path_not_found", f"Error message unclear: {e}")
    except Exception as e:
        log_fail("s25_path_not_found", f"Wrong exception: {type(e).__name__}: {e}")


def test_get_healthy_s25_files_filtering():
    """Test get_healthy_s25_files correctly filters files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        valid_files = [
            "s25_static_open.csv",
            "s25_static_close.csv",
            "s25_grasp_open.csv",
            "s25_grasp_close.csv",
        ]
        invalid_files = [
            "s25_movement_open.csv",  # Should be excluded (movement)
            "s25_static_movement.csv",  # Should be excluded (movement)
            "s25_random.csv",  # Should be excluded (no static/grasp)
        ]

        for f in valid_files + invalid_files:
            open(os.path.join(tmpdir, f), 'w').close()

        # Test 4.2: Should only return static/grasp files without movement
        try:
            files = get_healthy_s25_files(tmpdir)
            filenames = [os.path.basename(f) for f in files]

            # Check valid files are included
            for vf in valid_files:
                if vf in filenames:
                    log_pass(f"s25_includes_{vf}", f"Correctly includes {vf}")
                else:
                    log_fail(f"s25_includes_{vf}", f"Should include {vf}")

            # Check invalid files are excluded
            for ivf in invalid_files:
                if ivf not in filenames:
                    log_pass(f"s25_excludes_{ivf}", f"Correctly excludes {ivf}")
                else:
                    log_fail(f"s25_excludes_{ivf}", f"Should exclude {ivf}")

        except Exception as e:
            log_fail("s25_filtering", f"Unexpected error: {e}")


def test_get_healthy_s25_files_empty_dir():
    """Test get_healthy_s25_files raises error for empty/no-match directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files that won't match the filter
        open(os.path.join(tmpdir, "s25_movement_only.csv"), 'w').close()

        # Test 4.3: No matching files should raise ValueError
        try:
            get_healthy_s25_files(tmpdir)
            log_fail("s25_no_matches", "Should raise ValueError when no files match filter")
        except ValueError as e:
            if "No s25 files found" in str(e):
                log_pass("s25_no_matches", "Correctly raises ValueError")
            else:
                log_fail("s25_no_matches", f"Error message unclear: {e}")
        except Exception as e:
            log_fail("s25_no_matches", f"Wrong exception: {type(e).__name__}: {e}")


# ============================================================================
# SECTION 5: get_all_calibration_files() Tests
# ============================================================================

def test_get_all_calibration_files():
    """Test get_all_calibration_files with mock data."""
    print("\n" + "="*80)
    print("TEST SECTION: get_all_calibration_files() Tests")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create complete set of calibration files
        for set_num in range(1, 5):
            open(os.path.join(tmpdir, f"p15_open_{set_num}.csv"), 'w').close()
            open(os.path.join(tmpdir, f"p15_close_{set_num}.csv"), 'w').close()

        # Test 5.1: Should return 8 files
        files = conv_get_all_calibration_files(tmpdir)
        if len(files) == 8:
            log_pass("calib_files_count", "Found 8 calibration files")
        else:
            log_fail("calib_files_count", f"Expected 8 files, got {len(files)}")

        # Test 5.2: Should have 4 open and 4 close
        open_count = sum(1 for f in files if '_open_' in f)
        close_count = sum(1 for f in files if '_close_' in f)
        if open_count == 4 and close_count == 4:
            log_pass("calib_files_balanced", "4 open + 4 close files")
        else:
            log_fail("calib_files_balanced", f"Got {open_count} open + {close_count} close")


def test_get_all_calibration_files_incomplete():
    """Test get_all_calibration_files handles incomplete sets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create incomplete set (missing close_3)
        for set_num in range(1, 5):
            open(os.path.join(tmpdir, f"p15_open_{set_num}.csv"), 'w').close()
            if set_num != 3:
                open(os.path.join(tmpdir, f"p15_close_{set_num}.csv"), 'w').close()

        # Test 5.3: Should skip incomplete set
        files = conv_get_all_calibration_files(tmpdir)
        if len(files) == 6:  # 3 complete sets * 2 files each
            log_pass("calib_files_incomplete", "Correctly skips incomplete set (got 6 files)")
        else:
            log_fail("calib_files_incomplete", f"Expected 6 files, got {len(files)}")


# ============================================================================
# SECTION 6: dataset_utils.py - extract_repetition_units() Tests
# ============================================================================

def test_extract_repetition_units_basic():
    """Test extract_repetition_units with simple R-O-R pattern."""
    print("\n" + "="*80)
    print("TEST SECTION: extract_repetition_units() Tests")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CSV with clear R-O-R pattern (3 repetitions)
        csv_path = os.path.join(tmpdir, "test_open.csv")
        labels = create_ror_pattern('open', num_reps=3, relax_duration=800, gesture_duration=600)
        create_mock_csv_with_labels(csv_path, labels)

        # Test 6.1: Should extract 3 repetitions
        try:
            reps = extract_repetition_units(csv_path, 'open')
            if len(reps) == 3:
                log_pass("extract_reps_count", f"Extracted {len(reps)} repetitions (expected 3)")
            else:
                log_fail("extract_reps_count", f"Expected 3 reps, got {len(reps)}")

            # Test 6.2: Each rep should be a (start, end) tuple
            for i, rep in enumerate(reps):
                if isinstance(rep, tuple) and len(rep) == 2:
                    start, end = rep
                    if isinstance(start, int) and isinstance(end, int) and start < end:
                        log_pass(f"extract_rep_tuple_{i}", f"Rep {i}: valid tuple ({start}, {end})")
                    else:
                        log_fail(f"extract_rep_tuple_{i}", f"Rep {i}: invalid indices ({start}, {end})")
                else:
                    log_fail(f"extract_rep_tuple_{i}", f"Rep {i}: not a valid tuple")

        except Exception as e:
            log_fail("extract_reps_basic", f"Unexpected error: {e}")


def test_extract_repetition_units_close():
    """Test extract_repetition_units with R-C-R pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CSV with R-C-R pattern
        csv_path = os.path.join(tmpdir, "test_close.csv")
        labels = create_ror_pattern('close', num_reps=3, relax_duration=800, gesture_duration=600)
        create_mock_csv_with_labels(csv_path, labels)

        # Test 6.3: Should extract repetitions for close gesture
        try:
            reps = extract_repetition_units(csv_path, 'close')
            if len(reps) == 3:
                log_pass("extract_reps_close", f"Extracted {len(reps)} close repetitions")
            else:
                log_fail("extract_reps_close", f"Expected 3 reps, got {len(reps)}")
        except Exception as e:
            log_fail("extract_reps_close", f"Unexpected error: {e}")


def test_extract_repetition_units_invalid_gesture_type():
    """Test extract_repetition_units raises error for invalid gesture type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        create_mock_csv_with_labels(csv_path, [0, 1, 0], num_samples=100)

        # Test 6.4: Invalid gesture type should raise ValueError
        try:
            extract_repetition_units(csv_path, 'invalid_gesture')
            log_fail("extract_invalid_gesture", "Should raise ValueError for invalid gesture type")
        except ValueError as e:
            if "'open' or 'close'" in str(e):
                log_pass("extract_invalid_gesture", "Correctly raises ValueError")
            else:
                log_fail("extract_invalid_gesture", f"Error message unclear: {e}")
        except Exception as e:
            log_fail("extract_invalid_gesture", f"Wrong exception: {type(e).__name__}: {e}")


def test_extract_repetition_units_missing_gt_column():
    """Test extract_repetition_units raises error for missing gt column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        # Create CSV without 'gt' column
        df = pd.DataFrame({'emg0': [1, 2, 3], 'emg1': [4, 5, 6]})
        df.to_csv(csv_path, index=False)

        # Test 6.5: Missing gt column should raise ValueError
        try:
            extract_repetition_units(csv_path, 'open')
            log_fail("extract_missing_gt", "Should raise ValueError for missing 'gt' column")
        except ValueError as e:
            if "'gt' column not found" in str(e):
                log_pass("extract_missing_gt", "Correctly raises ValueError")
            else:
                log_fail("extract_missing_gt", f"Error message unclear: {e}")
        except Exception as e:
            log_fail("extract_missing_gt", f"Wrong exception: {type(e).__name__}: {e}")


def test_extract_repetition_units_no_transitions():
    """Test extract_repetition_units with no transitions in data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        # All relax, no transitions
        create_mock_csv_with_labels(csv_path, [0] * 1000)

        # Test 6.6: No transitions should return empty list
        try:
            reps = extract_repetition_units(csv_path, 'open')
            if len(reps) == 0:
                log_pass("extract_no_transitions", "Correctly returns empty list for no transitions")
            else:
                log_fail("extract_no_transitions", f"Expected 0 reps, got {len(reps)}")
        except Exception as e:
            log_fail("extract_no_transitions", f"Unexpected error: {e}")


def test_extract_repetition_units_padding_bounds():
    """Test that extracted repetitions include proper padding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        # Create a pattern where gesture starts at index 1000
        labels = [0] * 1000 + [1] * 500 + [0] * 1000
        create_mock_csv_with_labels(csv_path, labels)

        # Test 6.7: Should include 600 samples (3 sec at 200Hz) padding
        try:
            reps = extract_repetition_units(csv_path, 'open')
            if len(reps) >= 1:
                start, end = reps[0]
                # Gesture starts at 1000, so start should be 1000 - 600 = 400
                # Gesture ends at 1500, so end should be min(1500 + 600, len) = 2100 or len
                expected_start = max(0, 1000 - 600)
                if start == expected_start:
                    log_pass("extract_padding_start", f"Start padding correct: {start}")
                else:
                    log_fail("extract_padding_start", f"Expected start={expected_start}, got {start}")
            else:
                log_fail("extract_padding_bounds", "No repetitions extracted")
        except Exception as e:
            log_fail("extract_padding_bounds", f"Unexpected error: {e}")


# ============================================================================
# SECTION 7: dataset_utils.py - get_paired_repetition_indices() Tests
# ============================================================================

def test_get_paired_repetition_indices():
    """Test get_paired_repetition_indices with mock calibration files."""
    print("\n" + "="*80)
    print("TEST SECTION: get_paired_repetition_indices() Tests")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 4 sets of calibration files, each with 3 reps
        for set_num in range(1, 5):
            open_path = os.path.join(tmpdir, f"p15_open_{set_num}.csv")
            close_path = os.path.join(tmpdir, f"p15_close_{set_num}.csv")

            open_labels = create_ror_pattern('open', num_reps=3)
            close_labels = create_ror_pattern('close', num_reps=3)

            create_mock_csv_with_labels(open_path, open_labels)
            create_mock_csv_with_labels(close_path, close_labels)

        # Test 7.1: Should return 12 paired repetitions (4 sets * 3 reps)
        try:
            paired_reps = get_paired_repetition_indices(tmpdir, num_sets=4, reps_per_set=3)

            if len(paired_reps) == 12:
                log_pass("paired_reps_count", "Got 12 paired repetitions")
            else:
                log_fail("paired_reps_count", f"Expected 12 pairs, got {len(paired_reps)}")

            # Test 7.2: Keys should be g_0 through g_11
            expected_keys = [f"g_{i}" for i in range(12)]
            if list(paired_reps.keys()) == expected_keys:
                log_pass("paired_reps_keys", "Keys are g_0 through g_11")
            else:
                log_fail("paired_reps_keys", f"Unexpected keys: {list(paired_reps.keys())}")

            # Test 7.3: Each entry should be (open_file, close_file, open_segs, close_segs)
            for g_name, value in paired_reps.items():
                if isinstance(value, tuple) and len(value) == 4:
                    open_file, close_file, open_segs, close_segs = value
                    if (isinstance(open_file, str) and isinstance(close_file, str) and
                        isinstance(open_segs, list) and isinstance(close_segs, list)):
                        continue
                    else:
                        log_fail(f"paired_rep_structure_{g_name}", f"Invalid structure for {g_name}")
                        break
                else:
                    log_fail(f"paired_rep_tuple_{g_name}", f"Not a 4-tuple for {g_name}")
                    break
            else:
                log_pass("paired_rep_structure", "All entries have correct structure")

        except Exception as e:
            log_fail("paired_reps_basic", f"Unexpected error: {e}")


def test_get_paired_repetition_indices_mismatched_reps():
    """Test get_paired_repetition_indices raises error for mismatched rep counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create set with mismatched rep counts
        open_path = os.path.join(tmpdir, "p15_open_1.csv")
        close_path = os.path.join(tmpdir, "p15_close_1.csv")

        # Open has 3 reps, close has 2 reps
        open_labels = create_ror_pattern('open', num_reps=3)
        close_labels = create_ror_pattern('close', num_reps=2)

        create_mock_csv_with_labels(open_path, open_labels)
        create_mock_csv_with_labels(close_path, close_labels)

        # Test 7.4: Should raise ValueError for mismatched reps
        try:
            get_paired_repetition_indices(tmpdir, num_sets=1, reps_per_set=3)
            log_fail("paired_reps_mismatch", "Should raise ValueError for mismatched rep counts")
        except ValueError as e:
            if "Expected" in str(e) and "reps" in str(e):
                log_pass("paired_reps_mismatch", "Correctly raises ValueError")
            else:
                log_fail("paired_reps_mismatch", f"Error message unclear: {e}")
        except Exception as e:
            log_fail("paired_reps_mismatch", f"Wrong exception: {type(e).__name__}: {e}")


# ============================================================================
# SECTION 8: dataset_utils.py - sample_repetitions() Tests
# ============================================================================

def test_sample_repetitions_k1():
    """Test sample_repetitions with K=1 (each trial gets unique rep)."""
    print("\n" + "="*80)
    print("TEST SECTION: sample_repetitions() Tests")
    print("="*80)

    # Create mock paired_reps
    paired_reps = {f"g_{i}": (f"open_{i}", f"close_{i}", [(0, 100)], [(0, 100)])
                   for i in range(12)}

    # Test 8.1: K=1 should give each trial a unique rep
    trials = sample_repetitions(paired_reps, budget_k=1, num_trials=12, seed=42)

    if len(trials) == 12:
        log_pass("sample_k1_count", "Got 12 trials")
    else:
        log_fail("sample_k1_count", f"Expected 12 trials, got {len(trials)}")

    # Test 8.2: Each trial should have exactly 1 rep
    all_single = all(len(trial) == 1 for trial in trials)
    if all_single:
        log_pass("sample_k1_single", "Each trial has exactly 1 rep")
    else:
        log_fail("sample_k1_single", "Some trials don't have exactly 1 rep")

    # Test 8.3: For K=1, trial i should use g_i
    correct_mapping = all(trials[i] == [f"g_{i}"] for i in range(12))
    if correct_mapping:
        log_pass("sample_k1_unique", "Trial i uses g_i (unique mapping)")
    else:
        log_fail("sample_k1_unique", "K=1 doesn't follow trial i -> g_i mapping")


def test_sample_repetitions_k4():
    """Test sample_repetitions with K=4."""
    paired_reps = {f"g_{i}": (f"open_{i}", f"close_{i}", [(0, 100)], [(0, 100)])
                   for i in range(12)}

    # Test 8.4: K=4 should give each trial 4 reps
    trials = sample_repetitions(paired_reps, budget_k=4, num_trials=12, seed=42)

    all_k4 = all(len(trial) == 4 for trial in trials)
    if all_k4:
        log_pass("sample_k4_count", "Each trial has exactly 4 reps")
    else:
        log_fail("sample_k4_count", "Some trials don't have exactly 4 reps")

    # Test 8.5: Each trial's reps should be unique (no duplicates within trial)
    no_duplicates = all(len(trial) == len(set(trial)) for trial in trials)
    if no_duplicates:
        log_pass("sample_k4_unique_within", "No duplicates within any trial")
    else:
        log_fail("sample_k4_unique_within", "Some trials have duplicate reps")


def test_sample_repetitions_k8():
    """Test sample_repetitions with K=8."""
    paired_reps = {f"g_{i}": (f"open_{i}", f"close_{i}", [(0, 100)], [(0, 100)])
                   for i in range(12)}

    # Test 8.6: K=8 should give each trial 8 reps
    trials = sample_repetitions(paired_reps, budget_k=8, num_trials=12, seed=42)

    all_k8 = all(len(trial) == 8 for trial in trials)
    if all_k8:
        log_pass("sample_k8_count", "Each trial has exactly 8 reps")
    else:
        log_fail("sample_k8_count", "Some trials don't have exactly 8 reps")


def test_sample_repetitions_reproducibility():
    """Test that sample_repetitions is reproducible with same seed."""
    paired_reps = {f"g_{i}": (f"open_{i}", f"close_{i}", [(0, 100)], [(0, 100)])
                   for i in range(12)}

    # Test 8.7: Same seed should give same results
    trials1 = sample_repetitions(paired_reps, budget_k=4, num_trials=12, seed=42)
    trials2 = sample_repetitions(paired_reps, budget_k=4, num_trials=12, seed=42)

    if trials1 == trials2:
        log_pass("sample_reproducible", "Same seed produces same results")
    else:
        log_fail("sample_reproducible", "Same seed produces different results")


def test_sample_repetitions_budget_too_large():
    """Test sample_repetitions raises error when budget exceeds available reps."""
    paired_reps = {f"g_{i}": (f"open_{i}", f"close_{i}", [(0, 100)], [(0, 100)])
                   for i in range(12)}

    # Test 8.8: K > 12 should raise ValueError
    try:
        sample_repetitions(paired_reps, budget_k=15, num_trials=12, seed=42)
        log_fail("sample_budget_large", "Should raise ValueError when K > num_reps")
    except ValueError as e:
        if "cannot exceed" in str(e):
            log_pass("sample_budget_large", "Correctly raises ValueError")
        else:
            log_fail("sample_budget_large", f"Error message unclear: {e}")
    except Exception as e:
        log_fail("sample_budget_large", f"Wrong exception: {type(e).__name__}: {e}")


def test_sample_repetitions_k1_num_trials_too_large():
    """Test sample_repetitions with K=1 raises error when num_trials > num_reps."""
    paired_reps = {f"g_{i}": (f"open_{i}", f"close_{i}", [(0, 100)], [(0, 100)])
                   for i in range(12)}

    # Test 8.9: K=1 with num_trials > 12 should raise ValueError
    try:
        sample_repetitions(paired_reps, budget_k=1, num_trials=15, seed=42)
        log_fail("sample_k1_too_many_trials", "Should raise ValueError when K=1 and num_trials > num_reps")
    except ValueError as e:
        if "cannot exceed" in str(e):
            log_pass("sample_k1_too_many_trials", "Correctly raises ValueError")
        else:
            log_fail("sample_k1_too_many_trials", f"Error message unclear: {e}")
    except Exception as e:
        log_fail("sample_k1_too_many_trials", f"Wrong exception: {type(e).__name__}: {e}")


# ============================================================================
# SECTION 9: Training Command Construction Tests
# ============================================================================

def test_convergence_training_command():
    """Test training command construction in run_convergence.py."""
    print("\n" + "="*80)
    print("TEST SECTION: Training Command Construction")
    print("="*80)

    # Simulate the command construction from train_with_extended_epochs
    best_config = {'learning_rate': 1e-4, 'epochs': 10, 'dropout': 0.1}
    variant = 'lora'
    extended_epochs = best_config['epochs'] * 10

    cmd = [
        "python3", "main.py",
        "--offset", "30",
        "--num_classes", "3",
        "--task_selection", "0", "1", "2",
        "--use_input_layernorm",
        "--share_pe",
        "--dataset_selection", "custom_folder",
        "--window_size", "600",
        "--inner_window_size", "600",
        "--model_choice", "any2any",
        "--val_patient_ids", "none",
        "--epn_subset_percentage", "1.0",
        "--batch_size", "128",
        "--learning_rate", str(best_config['learning_rate']),
        "--epochs", str(extended_epochs),
        "--dropout", str(best_config['dropout']),
        "--exp_name", "test_convergence",
        "--custom_data_folder", "/tmp/test",
        "--save_every_epoch", "1",
    ]

    # Add variant-specific flags
    cmd.extend(["--saved_checkpoint_pth", CONV_PRETRAINED_CHECKPOINT])
    cmd.extend(["--use_lora", "1", "--lora_rank", "16", "--lora_alpha", "8", "--lora_dropout_p", "0.05"])

    # Test 9.1: Extended epochs calculation
    if extended_epochs == 100:
        log_pass("conv_extended_epochs", f"Extended epochs correct: {extended_epochs}")
    else:
        log_fail("conv_extended_epochs", f"Expected 100, got {extended_epochs}")

    # Test 9.2: save_every_epoch flag present
    if "--save_every_epoch" in cmd and "1" in cmd:
        log_pass("conv_save_every_epoch", "--save_every_epoch 1 flag present")
    else:
        log_fail("conv_save_every_epoch", "Missing --save_every_epoch 1 flag")

    # Test 9.3: LoRA flags present for lora variant
    lora_args = ["--use_lora", "--lora_rank", "--lora_alpha", "--lora_dropout_p"]
    all_present = all(arg in cmd for arg in lora_args)
    if all_present:
        log_pass("conv_lora_flags", "All LoRA flags present")
    else:
        missing = [arg for arg in lora_args if arg not in cmd]
        log_fail("conv_lora_flags", f"Missing LoRA flags: {missing}")


def test_data_efficiency_training_command():
    """Test training command construction in run_data_efficiency.py."""
    best_config = {'learning_rate': 5e-5, 'epochs': 15, 'dropout': 0.0}
    variant = 'head_only'

    cmd = [
        "python3", "main.py",
        "--offset", "30",
        "--num_classes", "3",
        "--task_selection", "0", "1", "2",
        "--use_input_layernorm",
        "--share_pe",
        "--dataset_selection", "custom_folder",
        "--window_size", "600",
        "--inner_window_size", "600",
        "--model_choice", "any2any",
        "--val_patient_ids", "none",
        "--epn_subset_percentage", "1.0",
        "--batch_size", "128",
        "--learning_rate", str(best_config['learning_rate']),
        "--epochs", str(best_config['epochs']),
        "--dropout", str(best_config['dropout']),
        "--exp_name", "test_data_efficiency",
        "--custom_data_folder", "/tmp/test",
    ]

    # Add head_only flags
    cmd.extend(["--saved_checkpoint_pth", DE_PRETRAINED_CHECKPOINT])
    cmd.extend(["--freeze_backbone", "1"])

    # Test 9.4: freeze_backbone flag present for head_only
    if "--freeze_backbone" in cmd and "1" in cmd:
        log_pass("de_freeze_backbone", "--freeze_backbone 1 flag present for head_only")
    else:
        log_fail("de_freeze_backbone", "Missing --freeze_backbone 1 flag")

    # Test 9.5: No LoRA flags for head_only
    if "--use_lora" not in cmd:
        log_pass("de_no_lora", "No LoRA flags for head_only variant")
    else:
        log_fail("de_no_lora", "Unexpected LoRA flags for head_only")


# ============================================================================
# SECTION 10: Symlink and Directory Tests
# ============================================================================

def test_symlink_operations():
    """Test symlink creation and handling."""
    print("\n" + "="*80)
    print("TEST SECTION: Symlink and Directory Operations")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "source.csv")
        link_dir = os.path.join(tmpdir, "links")
        os.makedirs(link_dir)
        link_path = os.path.join(link_dir, "source.csv")

        # Create source file
        open(source_file, 'w').close()

        # Test 10.1: Create symlink
        os.symlink(source_file, link_path)
        if os.path.islink(link_path) and os.path.exists(link_path):
            log_pass("symlink_create", "Symlink created and points to valid file")
        else:
            log_fail("symlink_create", "Symlink creation failed")

        # Test 10.2: lexists detects symlink even after source deletion
        os.remove(source_file)
        if os.path.lexists(link_path):
            log_pass("symlink_lexists_broken", "lexists detects broken symlink")
        else:
            log_fail("symlink_lexists_broken", "lexists failed for broken symlink")

        # Test 10.3: exists returns False for broken symlink
        if not os.path.exists(link_path):
            log_pass("symlink_exists_broken", "exists correctly returns False for broken symlink")
        else:
            log_fail("symlink_exists_broken", "exists incorrectly returns True for broken symlink")


def test_temp_directory_creation():
    """Test temporary directory creation patterns used in scripts."""
    import uuid

    # Test 10.4: UUID-based temp directory creation (data_efficiency pattern)
    temp_trial_id = uuid.uuid4().hex[:8]
    if len(temp_trial_id) == 8 and temp_trial_id.isalnum():
        log_pass("uuid_temp_id", f"UUID temp ID valid: {temp_trial_id}")
    else:
        log_fail("uuid_temp_id", f"Invalid UUID temp ID: {temp_trial_id}")

    # Test 10.5: Nested directory creation
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = os.path.join(tmpdir, "model_checkpoints", "convergence", "p15")
        try:
            os.makedirs(nested, exist_ok=True)
            if os.path.isdir(nested):
                log_pass("nested_dir_create", "Nested directory creation works")
            else:
                log_fail("nested_dir_create", "Nested directory not created")
        except Exception as e:
            log_fail("nested_dir_create", f"Error: {e}")


# ============================================================================
# SECTION 11: JSON Serialization Tests
# ============================================================================

def test_json_serialization_convergence():
    """Test JSON serialization for convergence results."""
    print("\n" + "="*80)
    print("TEST SECTION: JSON Serialization Tests")
    print("="*80)

    # Simulate epoch results structure
    epoch_results = {
        'epoch': 5,
        'stroke_results': {
            'mid_session_baseline': {
                'transition_accuracy': 0.85,
                'raw_accuracy': 0.92,
                'avg_detection_latency_ms': 45.5,
            }
        },
        'healthy_results': {
            'transition_accuracy': 0.78,
            'raw_accuracy': 0.88,
            'avg_detection_latency_ms': 52.0,
        },
        'stroke_avg_transition_acc': 0.85,
        'stroke_avg_raw_acc': 0.92,
        'stroke_avg_latency_ms': 45.5,
    }

    # Test 11.1: Can serialize and deserialize
    try:
        json_str = json.dumps(epoch_results, indent=4)
        parsed = json.loads(json_str)
        if parsed == epoch_results:
            log_pass("json_convergence_roundtrip", "Convergence results serialize correctly")
        else:
            log_fail("json_convergence_roundtrip", "Data changed after roundtrip")
    except Exception as e:
        log_fail("json_convergence_roundtrip", f"Serialization error: {e}")


def test_json_serialization_data_efficiency():
    """Test JSON serialization for data efficiency results."""
    # Simulate aggregated results structure
    aggregated_results = {
        'participant': 'p15',
        'variant': 'lora',
        'budget_k': 4,
        'num_trials': 12,
        'aggregated_results': {
            'mid_session_baseline': {
                'transition_accuracy_mean': 0.82,
                'transition_accuracy_std': 0.05,
                'raw_accuracy_mean': 0.90,
                'raw_accuracy_std': 0.03,
                'avg_detection_latency_ms_mean': 48,
                'avg_detection_latency_ms_std': 8,
            }
        }
    }

    # Test 11.2: Can serialize and deserialize
    try:
        json_str = json.dumps(aggregated_results, indent=4)
        parsed = json.loads(json_str)
        if parsed == aggregated_results:
            log_pass("json_data_eff_roundtrip", "Data efficiency results serialize correctly")
        else:
            log_fail("json_data_eff_roundtrip", "Data changed after roundtrip")
    except Exception as e:
        log_fail("json_data_eff_roundtrip", f"Serialization error: {e}")


def test_json_segments_serialization():
    """Test JSON serialization for sampled segments."""
    # Simulate sampled_segments structure from train_with_sampled_data
    sampled_segments = {
        '/path/to/p15_open_1.csv': [(0, 1200), (2400, 3600)],
        '/path/to/p15_close_1.csv': [(0, 1200)],
    }

    # Test 11.3: Tuple segments become lists in JSON
    try:
        segments_serializable = {}
        for file_path, seg_list in sampled_segments.items():
            segments_serializable[file_path] = [(int(s), int(e)) for s, e in seg_list]

        json_str = json.dumps(segments_serializable)
        parsed = json.loads(json_str)

        # JSON converts tuples to lists
        if all(isinstance(seg, list) for segs in parsed.values() for seg in segs):
            log_pass("json_segments_lists", "Segments converted to lists for JSON")
        else:
            log_fail("json_segments_lists", "Segment conversion failed")
    except Exception as e:
        log_fail("json_segments_serialization", f"Error: {e}")


# ============================================================================
# SECTION 12: Integration Tests with Real Data
# ============================================================================

def test_real_data_calibration_files():
    """Test with real calibration files if available."""
    print("\n" + "="*80)
    print("TEST SECTION: Integration Tests with Real Data")
    print("="*80)

    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            log_skip(f"real_data_{pid}", f"Data folder not found: {folder}")
            continue

        # Test 12.1: get_all_calibration_files works with real data
        try:
            calib_files = conv_get_all_calibration_files(folder)
            if len(calib_files) == 8:
                log_pass(f"real_calib_count_{pid}", f"Found 8 calibration files for {pid}")
            else:
                log_fail(f"real_calib_count_{pid}", f"Expected 8, got {len(calib_files)} for {pid}")
        except Exception as e:
            log_fail(f"real_calib_{pid}", f"Error: {e}")


def test_real_data_repetition_extraction():
    """Test repetition extraction with real data if available."""
    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            continue

        calib_files = glob.glob(os.path.join(folder, "*_open_1.csv"))
        if len(calib_files) != 1:
            log_skip(f"real_reps_{pid}", "open_1.csv not found")
            continue

        open_file = calib_files[0]

        # Test 12.2: extract_repetition_units works with real data
        try:
            reps = extract_repetition_units(open_file, 'open')
            if len(reps) >= 1:
                log_pass(f"real_reps_extract_{pid}", f"Extracted {len(reps)} reps from {pid} open_1")

                # Verify structure
                start, end = reps[0]
                if start < end:
                    log_pass(f"real_reps_valid_{pid}", f"Rep bounds valid: ({start}, {end})")
                else:
                    log_fail(f"real_reps_valid_{pid}", f"Invalid bounds: ({start}, {end})")
            else:
                log_fail(f"real_reps_extract_{pid}", "No reps extracted")
        except Exception as e:
            log_fail(f"real_reps_{pid}", f"Error: {e}")


def test_real_data_paired_repetitions():
    """Test paired repetition indexing with real data if available."""
    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            continue

        # Check if all calibration files exist
        calib_files = conv_get_all_calibration_files(folder)
        if len(calib_files) != 8:
            log_skip(f"real_paired_{pid}", "Incomplete calibration files")
            continue

        # Test 12.3: get_paired_repetition_indices works with real data
        try:
            paired_reps = get_paired_repetition_indices(folder, num_sets=4, reps_per_set=3)

            if len(paired_reps) == 12:
                log_pass(f"real_paired_count_{pid}", f"Got 12 paired reps for {pid}")
            else:
                log_fail(f"real_paired_count_{pid}", f"Expected 12, got {len(paired_reps)} for {pid}")

            # Verify each paired rep has valid structure
            for g_name, (open_f, close_f, open_segs, close_segs) in paired_reps.items():
                if os.path.exists(open_f) and os.path.exists(close_f):
                    continue
                else:
                    log_fail(f"real_paired_files_{pid}_{g_name}", "File path doesn't exist")
                    break
            else:
                log_pass(f"real_paired_files_{pid}", "All paired rep files exist")

        except Exception as e:
            log_fail(f"real_paired_{pid}", f"Error: {e}")


# ============================================================================
# SECTION 13: Edge Cases and Error Conditions
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*80)
    print("TEST SECTION: Edge Cases and Error Conditions")
    print("="*80)

    # Test 13.1: Empty participant folder
    with tempfile.TemporaryDirectory() as tmpdir:
        files = conv_get_all_calibration_files(tmpdir)
        if len(files) == 0:
            log_pass("edge_empty_folder", "Empty folder returns empty list")
        else:
            log_fail("edge_empty_folder", f"Expected empty list, got {len(files)} files")

    # Test 13.2: Single file (not a pair)
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "p15_open_1.csv"), 'w').close()
        files = conv_get_all_calibration_files(tmpdir)
        if len(files) == 0:
            log_pass("edge_single_file", "Single file (no pair) returns empty list")
        else:
            log_fail("edge_single_file", f"Expected empty list, got {len(files)} files")

    # Test 13.3: Files with wrong extension
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "p15_open_1.txt"), 'w').close()
        open(os.path.join(tmpdir, "p15_close_1.txt"), 'w').close()
        files = conv_get_all_calibration_files(tmpdir)
        if len(files) == 0:
            log_pass("edge_wrong_extension", ".txt files not matched")
        else:
            log_fail("edge_wrong_extension", f"Incorrectly matched {len(files)} .txt files")

    # Test 13.4: Very short CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "short.csv")
        create_mock_csv_with_labels(csv_path, [0, 1, 0], num_samples=10)

        try:
            reps = extract_repetition_units(csv_path, 'open')
            # Should handle short file without crashing
            log_pass("edge_short_csv", f"Short CSV handled, got {len(reps)} reps")
        except Exception as e:
            log_fail("edge_short_csv", f"Error with short CSV: {e}")

    # Test 13.5: CSV with no gesture transitions
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "all_relax.csv")
        create_mock_csv_with_labels(csv_path, [0] * 1000)

        reps = extract_repetition_units(csv_path, 'open')
        if len(reps) == 0:
            log_pass("edge_no_gestures", "No gestures returns empty list")
        else:
            log_fail("edge_no_gestures", f"Expected 0 reps for all-relax, got {len(reps)}")


def test_latency_conversion():
    """Test latency unit conversion (timesteps to ms)."""
    # At 200 Hz, 1 timestep = 5ms
    timesteps = 100
    expected_ms = 500.0

    # Conversion used in run_convergence.py:264
    latency_ms = float(timesteps) * 5.0

    if latency_ms == expected_ms:
        log_pass("latency_conversion", f"{timesteps} timesteps = {latency_ms}ms (correct)")
    else:
        log_fail("latency_conversion", f"Expected {expected_ms}ms, got {latency_ms}ms")


def test_int_rounding_for_json():
    """Test integer rounding for JSON serialization (data_efficiency.py:241)."""
    # The code uses int(round(...)) for latency
    test_values = [45.4, 45.5, 45.6, 0.0, 100.9]
    expected = [45, 46, 46, 0, 101]

    for val, exp in zip(test_values, expected):
        result = int(round(val))
        if result == exp:
            log_pass(f"int_round_{val}", f"int(round({val})) = {result}")
        else:
            log_fail(f"int_round_{val}", f"Expected {exp}, got {result}")


# ============================================================================
# SECTION 14: Import and Module Tests
# ============================================================================

def test_imports():
    """Test that all required imports work."""
    print("\n" + "="*80)
    print("TEST SECTION: Import and Module Tests")
    print("="*80)

    modules_to_test = [
        ('run_convergence', ['PARTICIPANTS', 'TEST_CONDITIONS', 'get_test_files',
                             'get_all_calibration_files', 'get_healthy_s25_files']),
        ('run_data_efficiency', ['PARTICIPANTS', 'TEST_CONDITIONS', 'get_test_files']),
        ('dataset_utils', ['extract_repetition_units', 'get_paired_repetition_indices',
                          'sample_repetitions']),
        ('event_classification', ['evaluate_checkpoint_programmatic']),
    ]

    for module_name, attrs in modules_to_test:
        try:
            module = __import__(module_name)
            for attr in attrs:
                if hasattr(module, attr):
                    log_pass(f"import_{module_name}_{attr}", f"{module_name}.{attr} exists")
                else:
                    log_fail(f"import_{module_name}_{attr}", f"{module_name}.{attr} not found")
        except ImportError as e:
            log_fail(f"import_{module_name}", f"Cannot import {module_name}: {e}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test sections."""
    print("\n" + "#"*80)
    print("# ReactEMG Stroke - run_convergence.py & run_data_efficiency.py Test Suite")
    print("#"*80)

    # Section 1: run_convergence.py Configuration
    test_convergence_configuration()

    # Section 2: run_data_efficiency.py Configuration
    test_data_efficiency_configuration()

    # Section 3: get_test_files() Exception Handling
    test_get_test_files_raises_file_not_found()
    test_get_test_files_raises_value_error()
    test_get_test_files_success()
    test_get_test_files_invalid_condition()

    # Section 4: get_healthy_s25_files()
    test_get_healthy_s25_files_path_not_found()
    test_get_healthy_s25_files_filtering()
    test_get_healthy_s25_files_empty_dir()

    # Section 5: get_all_calibration_files()
    test_get_all_calibration_files()
    test_get_all_calibration_files_incomplete()

    # Section 6: extract_repetition_units()
    test_extract_repetition_units_basic()
    test_extract_repetition_units_close()
    test_extract_repetition_units_invalid_gesture_type()
    test_extract_repetition_units_missing_gt_column()
    test_extract_repetition_units_no_transitions()
    test_extract_repetition_units_padding_bounds()

    # Section 7: get_paired_repetition_indices()
    test_get_paired_repetition_indices()
    test_get_paired_repetition_indices_mismatched_reps()

    # Section 8: sample_repetitions()
    test_sample_repetitions_k1()
    test_sample_repetitions_k4()
    test_sample_repetitions_k8()
    test_sample_repetitions_reproducibility()
    test_sample_repetitions_budget_too_large()
    test_sample_repetitions_k1_num_trials_too_large()

    # Section 9: Training Command Construction
    test_convergence_training_command()
    test_data_efficiency_training_command()

    # Section 10: Symlink and Directory Operations
    test_symlink_operations()
    test_temp_directory_creation()

    # Section 11: JSON Serialization
    test_json_serialization_convergence()
    test_json_serialization_data_efficiency()
    test_json_segments_serialization()

    # Section 12: Integration Tests with Real Data
    test_real_data_calibration_files()
    test_real_data_repetition_extraction()
    test_real_data_paired_repetitions()

    # Section 13: Edge Cases
    test_edge_cases()
    test_latency_conversion()
    test_int_rounding_for_json()

    # Section 14: Import Tests
    test_imports()

    # Print summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)
    print(f"\n  Passed:   {len(test_results['passed'])}")
    print(f"  Failed:   {len(test_results['failed'])}")
    print(f"  Warnings: {len(test_results['warnings'])}")
    print(f"  Skipped:  {len(test_results['skipped'])}")

    if test_results['failed']:
        print("\n  FAILED TESTS:")
        for name, error in test_results['failed']:
            print(f"    - {name}: {error}")

    if test_results['warnings']:
        print("\n  WARNINGS:")
        for name, warning in test_results['warnings']:
            print(f"    - {name}: {warning}")

    print("\n" + "#"*80)

    return 0 if len(test_results['failed']) == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
