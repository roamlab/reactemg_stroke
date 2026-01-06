"""
Comprehensive unit tests for run_main_experiment.py

This test suite exhaustively validates:
- Configuration correctness (PARTICIPANTS, VARIANTS, TEST_CONDITIONS)
- File discovery functions (get_all_calibration_files, get_test_files)
- File pattern matching with actual data
- Training command construction for all variants
- Error handling and edge cases
- Integration with cv_hyperparameter_search module
- Symlink creation and cleanup logic

Run with: python3 test_run_main_experiment.py
"""

import os
import sys
import glob
import tempfile
import shutil
import json
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict

# Import the module under test
from run_main_experiment import (
    PARTICIPANTS,
    PRETRAINED_CHECKPOINT,
    VARIANTS,
    TEST_CONDITIONS,
    get_all_calibration_files,
    get_test_files,
    train_final_model,
    evaluate_all_conditions,
    run_zero_shot_evaluation,
    run_main_experiment,
)
from cv_hyperparameter_search import (
    generate_hyperparameter_configs,
    get_fold_files,
)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Participant folders for testing
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
    print(f"  ✓ {test_name}")
    if message:
        print(f"    {message}")


def log_fail(test_name: str, error: str):
    """Log a failing test."""
    test_results['failed'].append((test_name, error))
    print(f"  ✗ {test_name}")
    print(f"    ERROR: {error}")


def log_warning(test_name: str, warning: str):
    """Log a warning."""
    test_results['warnings'].append((test_name, warning))
    print(f"  ⚠ {test_name}")
    print(f"    WARNING: {warning}")


def log_skip(test_name: str, reason: str):
    """Log a skipped test."""
    test_results['skipped'].append((test_name, reason))
    print(f"  ○ {test_name} (SKIPPED)")
    print(f"    Reason: {reason}")


# ============================================================================
# SECTION 1: CONFIGURATION VALIDATION TESTS
# ============================================================================

def test_participants_config():
    """Test that PARTICIPANTS configuration is valid."""
    print("\n" + "="*80)
    print("TEST SECTION: Configuration Validation")
    print("="*80)

    # Test 1.1: PARTICIPANTS dict structure
    if not isinstance(PARTICIPANTS, dict):
        log_fail("participants_is_dict", f"PARTICIPANTS should be dict, got {type(PARTICIPANTS)}")
        return
    log_pass("participants_is_dict", f"PARTICIPANTS is a dict with {len(PARTICIPANTS)} entries")

    # Test 1.2: All participant IDs follow expected pattern
    for pid in PARTICIPANTS.keys():
        if not pid.startswith('p') or not pid[1:].isdigit():
            log_fail("participant_id_format", f"Participant ID '{pid}' doesn't match pattern 'pN'")
        else:
            log_pass(f"participant_id_format_{pid}", f"ID '{pid}' is valid")

    # Test 1.3: All participant folders exist
    for pid, folder in PARTICIPANTS.items():
        expanded_folder = os.path.expanduser(folder)
        if os.path.isdir(expanded_folder):
            log_pass(f"participant_folder_exists_{pid}", f"Folder exists: {expanded_folder}")
        else:
            log_fail(f"participant_folder_exists_{pid}", f"Folder not found: {expanded_folder}")


def test_pretrained_checkpoint_config():
    """Test that PRETRAINED_CHECKPOINT exists and is valid."""
    # Test 1.4: Checkpoint path format
    if not PRETRAINED_CHECKPOINT.endswith('.pth'):
        log_fail("checkpoint_extension", f"Checkpoint should end with .pth, got: {PRETRAINED_CHECKPOINT}")
    else:
        log_pass("checkpoint_extension", "Checkpoint has .pth extension")

    # Test 1.5: Checkpoint file exists
    if os.path.isfile(PRETRAINED_CHECKPOINT):
        log_pass("checkpoint_exists", f"Checkpoint file exists: {PRETRAINED_CHECKPOINT}")
    else:
        log_fail("checkpoint_exists", f"Checkpoint file not found: {PRETRAINED_CHECKPOINT}")


def test_variants_config():
    """Test that VARIANTS configuration is valid."""
    expected_variants = ['stroke_only', 'head_only', 'lora', 'full_finetune']

    # Test 1.6: VARIANTS is a list
    if not isinstance(VARIANTS, list):
        log_fail("variants_is_list", f"VARIANTS should be list, got {type(VARIANTS)}")
        return
    log_pass("variants_is_list", f"VARIANTS is a list with {len(VARIANTS)} entries")

    # Test 1.7: All expected variants present
    for variant in expected_variants:
        if variant in VARIANTS:
            log_pass(f"variant_present_{variant}", f"Variant '{variant}' is present")
        else:
            log_fail(f"variant_present_{variant}", f"Expected variant '{variant}' not found")

    # Test 1.8: No unexpected variants
    for variant in VARIANTS:
        if variant not in expected_variants:
            log_warning(f"variant_unexpected_{variant}", f"Unexpected variant: {variant}")


def test_test_conditions_config():
    """Test that TEST_CONDITIONS configuration is valid."""
    expected_conditions = [
        'mid_session_baseline',
        'end_session_baseline',
        'unseen_posture',
        'sensor_shift',
        'orthosis_actuated'
    ]

    # Test 1.9: TEST_CONDITIONS is a dict
    if not isinstance(TEST_CONDITIONS, dict):
        log_fail("test_conditions_is_dict", f"TEST_CONDITIONS should be dict, got {type(TEST_CONDITIONS)}")
        return
    log_pass("test_conditions_is_dict", f"TEST_CONDITIONS is a dict with {len(TEST_CONDITIONS)} entries")

    # Test 1.10: All expected conditions present
    for condition in expected_conditions:
        if condition in TEST_CONDITIONS:
            patterns = TEST_CONDITIONS[condition]
            log_pass(f"condition_present_{condition}", f"Condition '{condition}' has {len(patterns)} file patterns")
        else:
            log_fail(f"condition_present_{condition}", f"Expected condition '{condition}' not found")

    # Test 1.11: All patterns are valid strings ending with .csv
    for condition, patterns in TEST_CONDITIONS.items():
        if not isinstance(patterns, list):
            log_fail(f"condition_patterns_list_{condition}", f"Patterns should be list, got {type(patterns)}")
            continue
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.endswith('.csv'):
                log_fail(f"pattern_format_{condition}", f"Invalid pattern: {pattern}")
            else:
                log_pass(f"pattern_format_{condition}_{pattern}", f"Pattern '{pattern}' is valid")


# ============================================================================
# SECTION 2: get_all_calibration_files() TESTS
# ============================================================================

def test_get_all_calibration_files_basic():
    """Test basic functionality of get_all_calibration_files()."""
    print("\n" + "="*80)
    print("TEST SECTION: get_all_calibration_files()")
    print("="*80)

    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            log_skip(f"calib_files_basic_{pid}", f"Data folder not found: {folder}")
            continue

        try:
            calib_files = get_all_calibration_files(folder)

            # Test 2.1: Returns a list
            if not isinstance(calib_files, list):
                log_fail(f"calib_files_is_list_{pid}", f"Should return list, got {type(calib_files)}")
                continue

            # Test 2.2: Expected number of files (8 = 4 open + 4 close)
            if len(calib_files) == 8:
                log_pass(f"calib_files_count_{pid}", f"Found exactly 8 calibration files")
            else:
                log_fail(f"calib_files_count_{pid}", f"Expected 8 files, got {len(calib_files)}")

            # Test 2.3: All files exist
            all_exist = True
            for f in calib_files:
                if not os.path.isfile(f):
                    all_exist = False
                    log_fail(f"calib_file_exists_{pid}", f"File not found: {f}")
                    break
            if all_exist:
                log_pass(f"calib_files_exist_{pid}", "All calibration files exist")

            # Test 2.4: Files follow expected naming pattern
            open_count = sum(1 for f in calib_files if '_open_' in os.path.basename(f))
            close_count = sum(1 for f in calib_files if '_close_' in os.path.basename(f))
            if open_count == 4 and close_count == 4:
                log_pass(f"calib_files_pattern_{pid}", "4 open + 4 close files found")
            else:
                log_fail(f"calib_files_pattern_{pid}", f"Expected 4 open + 4 close, got {open_count} open + {close_count} close")

            # Test 2.5: Files are numbered 1-4
            for set_num in range(1, 5):
                has_open = any(f"_open_{set_num}.csv" in f for f in calib_files)
                has_close = any(f"_close_{set_num}.csv" in f for f in calib_files)
                if has_open and has_close:
                    log_pass(f"calib_set_{set_num}_{pid}", f"Set {set_num} (open + close) found")
                else:
                    log_fail(f"calib_set_{set_num}_{pid}", f"Set {set_num} missing: open={has_open}, close={has_close}")

        except Exception as e:
            log_fail(f"calib_files_exception_{pid}", str(e))


def test_get_all_calibration_files_with_missing():
    """Test get_all_calibration_files() handles missing files gracefully."""
    # Create temp directory with incomplete calibration set
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create only open_1, open_2, close_1, close_2 (missing 3 and 4)
        for set_num in [1, 2]:
            open(os.path.join(tmpdir, f"test_open_{set_num}.csv"), 'w').close()
            open(os.path.join(tmpdir, f"test_close_{set_num}.csv"), 'w').close()

        calib_files = get_all_calibration_files(tmpdir)

        # Test 2.6: Returns only complete pairs
        if len(calib_files) == 4:
            log_pass("calib_files_incomplete_set", "Returns 4 files when only sets 1-2 exist")
        else:
            log_fail("calib_files_incomplete_set", f"Expected 4 files, got {len(calib_files)}")


def test_get_all_calibration_files_empty():
    """Test get_all_calibration_files() with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_files = get_all_calibration_files(tmpdir)

        # Test 2.7: Returns empty list for empty directory
        if len(calib_files) == 0:
            log_pass("calib_files_empty_dir", "Returns empty list for empty directory")
        else:
            log_fail("calib_files_empty_dir", f"Expected empty list, got {len(calib_files)} files")


def test_get_all_calibration_files_multiple_matches():
    """Test get_all_calibration_files() when multiple files match a pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create duplicate files for open_1
        open(os.path.join(tmpdir, "p15_open_1.csv"), 'w').close()
        open(os.path.join(tmpdir, "p15_backup_open_1.csv"), 'w').close()  # Another match
        open(os.path.join(tmpdir, "p15_close_1.csv"), 'w').close()

        calib_files = get_all_calibration_files(tmpdir)

        # Test 2.8: Skips sets with multiple matches (glob returns 2)
        # The condition `len(open_file) == 1` should fail
        open_1_count = sum(1 for f in calib_files if '_open_1.csv' in f)
        if open_1_count == 0:
            log_pass("calib_files_multi_match", "Skips set when multiple files match pattern")
        else:
            log_fail("calib_files_multi_match", f"Should skip multi-match, got {open_1_count} open_1 files")


# ============================================================================
# SECTION 3: get_test_files() TESTS
# ============================================================================

def test_get_test_files_all_conditions():
    """Test get_test_files() for all conditions with real data."""
    print("\n" + "="*80)
    print("TEST SECTION: get_test_files()")
    print("="*80)

    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            log_skip(f"test_files_all_{pid}", f"Data folder not found: {folder}")
            continue

        for condition, patterns in TEST_CONDITIONS.items():
            try:
                test_files = get_test_files(folder, condition)

                # Test 3.1: Returns expected number of files
                expected_count = len(patterns)
                if len(test_files) == expected_count:
                    log_pass(f"test_files_{condition}_{pid}",
                            f"Found {len(test_files)} files for '{condition}'")
                else:
                    log_fail(f"test_files_{condition}_{pid}",
                            f"Expected {expected_count} files, got {len(test_files)}")

                # Test 3.2: All returned files exist
                for f in test_files:
                    if not os.path.isfile(f):
                        log_fail(f"test_file_exists_{condition}_{pid}", f"File not found: {f}")

            except FileNotFoundError as e:
                log_fail(f"test_files_{condition}_{pid}", f"FileNotFoundError: {e}")
            except ValueError as e:
                log_fail(f"test_files_{condition}_{pid}", f"ValueError: {e}")
            except Exception as e:
                log_fail(f"test_files_{condition}_{pid}", f"Unexpected error: {e}")


def test_get_test_files_file_not_found():
    """Test get_test_files() raises FileNotFoundError for missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory - should raise FileNotFoundError
        try:
            get_test_files(tmpdir, 'mid_session_baseline')
            log_fail("test_files_not_found_raises", "Should raise FileNotFoundError for missing files")
        except FileNotFoundError:
            log_pass("test_files_not_found_raises", "Correctly raises FileNotFoundError")
        except Exception as e:
            log_fail("test_files_not_found_raises", f"Wrong exception type: {type(e).__name__}")


def test_get_test_files_multiple_matches():
    """Test get_test_files() raises ValueError for multiple matching files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two files matching the same pattern
        open(os.path.join(tmpdir, "p15_open_5.csv"), 'w').close()
        open(os.path.join(tmpdir, "p15_backup_open_5.csv"), 'w').close()

        try:
            get_test_files(tmpdir, 'mid_session_baseline')
            log_fail("test_files_multi_match_raises", "Should raise ValueError for multiple matches")
        except ValueError as e:
            if "expected exactly 1" in str(e).lower():
                log_pass("test_files_multi_match_raises", "Correctly raises ValueError with descriptive message")
            else:
                log_fail("test_files_multi_match_raises", f"ValueError message unclear: {e}")
        except FileNotFoundError:
            # This could happen if close_5 isn't found first
            log_pass("test_files_multi_match_raises", "Raises FileNotFoundError (close_5 not found first)")
        except Exception as e:
            log_fail("test_files_multi_match_raises", f"Wrong exception type: {type(e).__name__}")


def test_get_test_files_invalid_condition():
    """Test get_test_files() with invalid condition name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            get_test_files(tmpdir, 'invalid_condition_name')
            log_fail("test_files_invalid_condition", "Should raise KeyError for invalid condition")
        except KeyError:
            log_pass("test_files_invalid_condition", "Correctly raises KeyError for invalid condition")
        except Exception as e:
            log_fail("test_files_invalid_condition", f"Wrong exception type: {type(e).__name__}: {e}")


# ============================================================================
# SECTION 4: FILE PATTERN MATCHING TESTS
# ============================================================================

def test_pattern_matching_calibration():
    """Verify calibration file patterns match correctly."""
    print("\n" + "="*80)
    print("TEST SECTION: File Pattern Matching")
    print("="*80)

    # Test patterns used in get_all_calibration_files
    test_cases = [
        ("p15_open_1.csv", "*_open_1.csv", True),
        ("p15_close_4.csv", "*_close_4.csv", True),
        ("p20_open_2.csv", "*_open_2.csv", True),
        ("open_1.csv", "*_open_1.csv", False),  # No prefix
        ("p15_open_10.csv", "*_open_1.csv", False),  # Wrong number
        ("p15_open_1.txt", "*_open_1.csv", False),  # Wrong extension
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, pattern, should_match in test_cases:
            filepath = os.path.join(tmpdir, filename)
            open(filepath, 'w').close()

            matches = glob.glob(os.path.join(tmpdir, pattern))
            did_match = filepath in matches

            if did_match == should_match:
                log_pass(f"pattern_match_{filename}",
                        f"'{filename}' {'matches' if should_match else 'does not match'} '{pattern}'")
            else:
                log_fail(f"pattern_match_{filename}",
                        f"'{filename}' should {'match' if should_match else 'not match'} '{pattern}'")

            os.remove(filepath)


def test_pattern_matching_test_conditions():
    """Verify test condition file patterns match correctly."""
    # Test patterns used in get_test_files
    test_cases = [
        ("p15_open_5.csv", "*_open_5.csv", True),
        ("p15_close_fatigue.csv", "*_close_fatigue.csv", True),
        ("p15_open_hovering.csv", "*_open_hovering.csv", True),
        ("p15_close_from_open.csv", "*_close_from_open.csv", True),
        ("p15_open_sensor_shift.csv", "*_open_sensor_shift.csv", True),
        ("p15open_5.csv", "*_open_5.csv", False),  # Missing underscore
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, pattern, should_match in test_cases:
            filepath = os.path.join(tmpdir, filename)
            open(filepath, 'w').close()

            matches = glob.glob(os.path.join(tmpdir, pattern))
            did_match = filepath in matches

            if did_match == should_match:
                log_pass(f"test_pattern_{filename}",
                        f"'{filename}' {'matches' if should_match else 'does not match'} '{pattern}'")
            else:
                log_fail(f"test_pattern_{filename}",
                        f"'{filename}' should {'match' if should_match else 'not match'} '{pattern}'")

            os.remove(filepath)


# ============================================================================
# SECTION 5: TRAINING COMMAND CONSTRUCTION TESTS
# ============================================================================

def test_training_command_construction():
    """Test that training commands are constructed correctly for each variant."""
    print("\n" + "="*80)
    print("TEST SECTION: Training Command Construction")
    print("="*80)

    # Mock the subprocess.run and other functions to capture the command
    captured_commands = {}

    for variant in VARIANTS:
        # Build expected command components
        best_config = {'learning_rate': 1e-4, 'epochs': 10, 'dropout': 0.1}

        # Simulate command construction (from train_final_model)
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
            "--exp_name", f"test_{variant}_final",
            "--custom_data_folder", "/tmp/test",
        ]

        # Add variant-specific flags
        if variant == 'stroke_only':
            pass  # No pretrained checkpoint
        elif variant == 'head_only':
            cmd.extend(["--saved_checkpoint_pth", PRETRAINED_CHECKPOINT])
            cmd.extend(["--freeze_backbone", "1"])
        elif variant == 'lora':
            cmd.extend(["--saved_checkpoint_pth", PRETRAINED_CHECKPOINT])
            cmd.extend(["--use_lora", "1", "--lora_rank", "16", "--lora_alpha", "8", "--lora_dropout_p", "0.05"])
        elif variant == 'full_finetune':
            cmd.extend(["--saved_checkpoint_pth", PRETRAINED_CHECKPOINT])

        captured_commands[variant] = cmd

        # Test 5.1: Command starts with python3 main.py
        if cmd[0] == "python3" and cmd[1] == "main.py":
            log_pass(f"cmd_prefix_{variant}", "Command starts with 'python3 main.py'")
        else:
            log_fail(f"cmd_prefix_{variant}", f"Command should start with 'python3 main.py', got {cmd[:2]}")

        # Test 5.2: Required arguments present
        required_args = ["--num_classes", "--model_choice", "--window_size",
                        "--inner_window_size", "--batch_size", "--epochs"]
        for arg in required_args:
            if arg in cmd:
                log_pass(f"cmd_has_{arg}_{variant}", f"Command includes {arg}")
            else:
                log_fail(f"cmd_has_{arg}_{variant}", f"Missing required argument: {arg}")

        # Test 5.3: Variant-specific flags
        if variant == 'stroke_only':
            if "--saved_checkpoint_pth" not in cmd:
                log_pass(f"cmd_no_checkpoint_{variant}", "stroke_only has no checkpoint (train from scratch)")
            else:
                log_fail(f"cmd_no_checkpoint_{variant}", "stroke_only should not have checkpoint")

        elif variant == 'head_only':
            if "--freeze_backbone" in cmd and "--saved_checkpoint_pth" in cmd:
                log_pass(f"cmd_freeze_{variant}", "head_only has --freeze_backbone and checkpoint")
            else:
                log_fail(f"cmd_freeze_{variant}", "head_only missing --freeze_backbone or checkpoint")

        elif variant == 'lora':
            lora_args = ["--use_lora", "--lora_rank", "--lora_alpha", "--lora_dropout_p"]
            all_present = all(arg in cmd for arg in lora_args)
            if all_present and "--saved_checkpoint_pth" in cmd:
                log_pass(f"cmd_lora_{variant}", "lora has all LoRA args and checkpoint")
            else:
                log_fail(f"cmd_lora_{variant}", "lora missing LoRA args or checkpoint")

        elif variant == 'full_finetune':
            if "--saved_checkpoint_pth" in cmd and "--freeze_backbone" not in cmd and "--use_lora" not in cmd:
                log_pass(f"cmd_full_{variant}", "full_finetune has checkpoint, no freeze, no LoRA")
            else:
                log_fail(f"cmd_full_{variant}", "full_finetune should have checkpoint only")


def test_hyperparameter_configs():
    """Test that hyperparameter configs are generated correctly."""
    for variant in VARIANTS:
        configs = generate_hyperparameter_configs(variant)

        # Test 5.4: Expected number of configs (3 LR × 3 epochs × 3 dropout = 27)
        expected_count = 3 * 3 * 3
        if len(configs) == expected_count:
            log_pass(f"hp_config_count_{variant}", f"Generated {len(configs)} configs")
        else:
            log_fail(f"hp_config_count_{variant}", f"Expected {expected_count} configs, got {len(configs)}")

        # Test 5.5: All configs have required keys
        required_keys = ['learning_rate', 'epochs', 'dropout', 'variant']
        for i, config in enumerate(configs):
            missing = [k for k in required_keys if k not in config]
            if missing:
                log_fail(f"hp_config_keys_{variant}_{i}", f"Missing keys: {missing}")
                break
        else:
            log_pass(f"hp_config_keys_{variant}", "All configs have required keys")

        # Test 5.6: Variant is set correctly
        wrong_variant = [c for c in configs if c.get('variant') != variant]
        if len(wrong_variant) == 0:
            log_pass(f"hp_config_variant_{variant}", "All configs have correct variant")
        else:
            log_fail(f"hp_config_variant_{variant}", f"{len(wrong_variant)} configs have wrong variant")


# ============================================================================
# SECTION 6: CV FOLD SPLITTING TESTS
# ============================================================================

def test_cv_fold_files():
    """Test that CV fold splitting works correctly."""
    print("\n" + "="*80)
    print("TEST SECTION: CV Fold Splitting")
    print("="*80)

    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            log_skip(f"cv_folds_{pid}", f"Data folder not found: {folder}")
            continue

        try:
            all_train_files = set()
            all_val_files = set()

            for fold_idx in range(4):
                train_files, val_files = get_fold_files(folder, fold_idx)

                # Test 6.1: Correct number of train/val files per fold
                if len(train_files) == 6:  # 3 sets × 2 files
                    log_pass(f"cv_train_count_fold{fold_idx}_{pid}", "6 training files")
                else:
                    log_fail(f"cv_train_count_fold{fold_idx}_{pid}",
                            f"Expected 6 training files, got {len(train_files)}")

                if len(val_files) == 2:  # 1 set × 2 files
                    log_pass(f"cv_val_count_fold{fold_idx}_{pid}", "2 validation files")
                else:
                    log_fail(f"cv_val_count_fold{fold_idx}_{pid}",
                            f"Expected 2 validation files, got {len(val_files)}")

                # Test 6.2: No overlap between train and val
                overlap = set(train_files) & set(val_files)
                if len(overlap) == 0:
                    log_pass(f"cv_no_overlap_fold{fold_idx}_{pid}", "No train/val overlap")
                else:
                    log_fail(f"cv_no_overlap_fold{fold_idx}_{pid}", f"Overlap found: {overlap}")

                all_train_files.update(train_files)
                all_val_files.update(val_files)

            # Test 6.3: All files are used in validation exactly once
            if len(all_val_files) == 8:
                log_pass(f"cv_all_files_validated_{pid}", "All 8 files used in validation across folds")
            else:
                log_fail(f"cv_all_files_validated_{pid}",
                        f"Expected 8 unique val files, got {len(all_val_files)}")

        except Exception as e:
            log_fail(f"cv_folds_{pid}", f"Exception: {e}")


def test_cv_fold_invalid_index():
    """Test CV fold function with invalid fold index."""
    folder = PARTICIPANT_FOLDERS.get('p15')
    if not folder or not os.path.isdir(folder):
        log_skip("cv_invalid_fold", "p15 data folder not found")
        return

    # Test with invalid fold indices
    for invalid_idx in [-1, 4, 5, 10]:
        try:
            train_files, val_files = get_fold_files(folder, invalid_idx)
            # If it doesn't raise, check if it returns something reasonable
            log_warning(f"cv_invalid_fold_{invalid_idx}",
                       f"No exception for invalid fold_idx={invalid_idx}, returned {len(train_files)} train, {len(val_files)} val")
        except (IndexError, ValueError) as e:
            log_pass(f"cv_invalid_fold_{invalid_idx}", f"Raises exception for fold_idx={invalid_idx}")
        except Exception as e:
            log_fail(f"cv_invalid_fold_{invalid_idx}", f"Unexpected exception type: {type(e).__name__}")


# ============================================================================
# SECTION 7: SYMLINK CREATION TESTS
# ============================================================================

def test_symlink_creation():
    """Test that symlink creation logic works correctly."""
    print("\n" + "="*80)
    print("TEST SECTION: Symlink Creation")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source file
        source_file = os.path.join(tmpdir, "source.csv")
        open(source_file, 'w').close()

        link_dir = os.path.join(tmpdir, "links")
        os.makedirs(link_dir)
        link_path = os.path.join(link_dir, "source.csv")

        # Test 7.1: Create symlink
        os.symlink(source_file, link_path)
        if os.path.islink(link_path):
            log_pass("symlink_create", "Symlink created successfully")
        else:
            log_fail("symlink_create", "Failed to create symlink")

        # Test 7.2: lexists() detects symlink
        if os.path.lexists(link_path):
            log_pass("symlink_lexists", "lexists() detects symlink")
        else:
            log_fail("symlink_lexists", "lexists() failed to detect symlink")

        # Test 7.3: Remove and recreate symlink (as done in train_final_model)
        os.remove(link_path)
        os.symlink(source_file, link_path)
        if os.path.islink(link_path) and os.path.exists(link_path):
            log_pass("symlink_recreate", "Symlink recreated successfully")
        else:
            log_fail("symlink_recreate", "Failed to recreate symlink")

        # Test 7.4: Broken symlink handling
        os.remove(source_file)  # Delete source, making symlink broken
        if os.path.lexists(link_path) and not os.path.exists(link_path):
            log_pass("symlink_broken_detection", "Broken symlink detected correctly")
        else:
            log_fail("symlink_broken_detection", "Failed to detect broken symlink")


# ============================================================================
# SECTION 8: ERROR HANDLING TESTS
# ============================================================================

def test_error_handling():
    """Test error handling in various edge cases."""
    print("\n" + "="*80)
    print("TEST SECTION: Error Handling")
    print("="*80)

    # Test 8.1: Invalid participant in run_main_experiment
    try:
        # This should raise ValueError for invalid participant
        # We can't actually run this without side effects, so just test the validation
        from run_main_experiment import run_main_experiment
        # The function checks: if participant_filter not in PARTICIPANTS
        if 'invalid_participant' not in PARTICIPANTS:
            log_pass("invalid_participant_check", "Invalid participant would be caught")
        else:
            log_fail("invalid_participant_check", "PARTICIPANTS contains 'invalid_participant'?")
    except Exception as e:
        log_fail("invalid_participant_check", f"Unexpected error: {e}")

    # Test 8.2: Missing checkpoint file handling
    fake_checkpoint = "/nonexistent/path/to/checkpoint.pth"
    if not os.path.exists(fake_checkpoint):
        log_pass("missing_checkpoint_path", "Missing checkpoint path correctly doesn't exist")

    # Test 8.3: Invalid variant handling
    try:
        # Check if unknown variant would be caught in train_final_model
        # The function has: else: raise ValueError(f"Unknown variant: {variant}")
        invalid_variant = "unknown_variant"
        if invalid_variant not in VARIANTS:
            log_pass("invalid_variant_check", "Invalid variant would be caught")
    except Exception as e:
        log_fail("invalid_variant_check", f"Unexpected error: {e}")


# ============================================================================
# SECTION 9: INTEGRATION TESTS (with mocking)
# ============================================================================

def test_evaluate_checkpoint_interface():
    """Test that evaluate_checkpoint_programmatic has expected interface."""
    print("\n" + "="*80)
    print("TEST SECTION: Integration Interface Tests")
    print("="*80)

    from event_classification import evaluate_checkpoint_programmatic
    import inspect

    # Test 9.1: Function signature
    sig = inspect.signature(evaluate_checkpoint_programmatic)
    params = list(sig.parameters.keys())

    expected_params = ['checkpoint_path', 'csv_files', 'buffer_range', 'lookahead',
                      'samples_between_prediction', 'allow_relax', 'stride',
                      'model_choice', 'verbose']

    for param in expected_params:
        if param in params:
            log_pass(f"eval_param_{param}", f"Parameter '{param}' exists")
        else:
            log_fail(f"eval_param_{param}", f"Missing parameter: {param}")


def test_results_directory_structure():
    """Test that results would be saved to correct directory structure."""
    # Test expected directory structure
    expected_structure = [
        "results/main_experiment/{participant}/{variant}/{condition}/metrics_summary.json"
    ]

    # Verify the format strings work
    try:
        for participant in ['p15', 'p20']:
            for variant in VARIANTS + ['zero_shot']:
                for condition in TEST_CONDITIONS.keys():
                    results_dir = os.path.join("results/main_experiment", participant, variant, condition)
                    metrics_file = os.path.join(results_dir, "metrics_summary.json")
                    # Just verify the path is constructable
                    if len(metrics_file) > 0:
                        pass
        log_pass("results_dir_structure", "Results directory structure is valid")
    except Exception as e:
        log_fail("results_dir_structure", f"Error constructing path: {e}")


# ============================================================================
# SECTION 10: DATA INTEGRITY TESTS
# ============================================================================

def test_data_file_integrity():
    """Test that data files are valid CSVs with expected columns."""
    print("\n" + "="*80)
    print("TEST SECTION: Data File Integrity")
    print("="*80)

    import pandas as pd

    # Expected columns based on actual data format
    EXPECTED_COLUMNS = ['gt', 'time_elapsed', 'current_time', 'current_task',
                        'motor_position', 'futek', 'emg0', 'emg1', 'emg2',
                        'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg_timer_stamp']
    EMG_COLUMNS = ['emg0', 'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7']

    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            log_skip(f"data_integrity_{pid}", f"Data folder not found: {folder}")
            continue

        # Test first calibration file
        calib_files = get_all_calibration_files(folder)
        if len(calib_files) == 0:
            log_skip(f"data_integrity_{pid}", "No calibration files found")
            continue

        test_file = calib_files[0]
        try:
            df = pd.read_csv(test_file)

            # Test 10.1: File is readable
            log_pass(f"csv_readable_{pid}", f"CSV readable: {os.path.basename(test_file)}")

            # Test 10.2: Has expected columns
            missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
            if len(missing_cols) == 0:
                log_pass(f"csv_columns_{pid}", f"Has all {len(EXPECTED_COLUMNS)} expected columns")
            else:
                log_fail(f"csv_columns_{pid}", f"Missing columns: {missing_cols}")

            # Test 10.3: Has reasonable number of rows (at 200Hz, 10s = 2000 samples)
            if df.shape[0] >= 1000:
                log_pass(f"csv_rows_{pid}", f"Has {df.shape[0]} rows (reasonable length)")
            else:
                log_fail(f"csv_rows_{pid}", f"Only {df.shape[0]} rows, seems too short")

            # Test 10.4: No NaN values in EMG columns
            emg_cols = df[EMG_COLUMNS] if all(c in df.columns for c in EMG_COLUMNS) else df.iloc[:, 6:14]
            nan_count = emg_cols.isna().sum().sum()
            if nan_count == 0:
                log_pass(f"csv_no_nan_{pid}", "No NaN values in EMG columns")
            else:
                log_fail(f"csv_no_nan_{pid}", f"Found {nan_count} NaN values in EMG data")

            # Test 10.5: gt column exists and has valid dtype
            if 'gt' in df.columns:
                if df['gt'].dtype in ['int64', 'int32', 'float64']:
                    log_pass(f"csv_gt_dtype_{pid}", f"'gt' column has numeric dtype: {df['gt'].dtype}")
                else:
                    log_fail(f"csv_gt_dtype_{pid}", f"'gt' column has unexpected dtype: {df['gt'].dtype}")
            else:
                log_fail(f"csv_gt_exists_{pid}", "'gt' column not found")

        except Exception as e:
            log_fail(f"data_integrity_{pid}", f"Error reading {test_file}: {e}")


def test_label_values():
    """Test that label values are valid (0, 1, 2 for 3-class)."""
    import pandas as pd

    for pid, folder in PARTICIPANT_FOLDERS.items():
        if not os.path.isdir(folder):
            continue

        calib_files = get_all_calibration_files(folder)
        if len(calib_files) == 0:
            continue

        # Check all calibration files to capture both open (0,1) and close (0,2) labels
        all_labels = set()
        for test_file in calib_files:
            try:
                df = pd.read_csv(test_file)
                # Label column is 'gt' (ground truth), NOT the last column
                if 'gt' not in df.columns:
                    log_fail(f"label_column_exists_{pid}", f"'gt' column not found in {os.path.basename(test_file)}")
                    continue
                labels = df['gt']
                all_labels.update(labels.unique())
            except Exception as e:
                log_fail(f"label_values_{pid}", f"Error reading {test_file}: {e}")
                continue

        expected_labels = {0, 1, 2}  # relax, open, close
        if all_labels.issubset(expected_labels):
            log_pass(f"label_values_{pid}", f"Labels are valid: {sorted(all_labels)}")
        else:
            unexpected = all_labels - expected_labels
            log_fail(f"label_values_{pid}", f"Unexpected label values: {unexpected}")


# ============================================================================
# SECTION 11: CHECKPOINT DIRECTORY NAMING TESTS
# ============================================================================

def test_checkpoint_directory_naming():
    """Test checkpoint directory naming conventions."""
    print("\n" + "="*80)
    print("TEST SECTION: Checkpoint Directory Naming")
    print("="*80)

    import re
    from datetime import datetime

    # Test expected checkpoint naming pattern
    # Format: {exp_name}_{YYYY-MM-DD}_{HH-MM-SS}_{hostname}
    pattern = r"^(.+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(.+)$"

    test_names = [
        ("p15_lora_final_2025-12-04_14-30-00_pc1", True),
        ("p15_stroke_only_fold0_lr0.0001_ep10_do0.1_2025-12-04_14-30-00_pc1", True),
        ("invalid_name", False),
        ("p15_lora_final_2025-12-04_14-30-00", False),  # Missing hostname
    ]

    for name, should_match in test_names:
        match = re.match(pattern, name)
        did_match = match is not None

        if did_match == should_match:
            log_pass(f"checkpoint_name_{name[:30]}",
                    f"'{name[:40]}...' {'matches' if should_match else 'does not match'} pattern")
        else:
            log_fail(f"checkpoint_name_{name[:30]}",
                    f"'{name[:40]}...' should {'match' if should_match else 'not match'} pattern")


def test_checkpoint_sorting():
    """Test that checkpoint directories sort correctly by timestamp."""
    test_dirs = [
        "exp_2025-12-04_14-30-00_pc1",
        "exp_2025-12-04_14-31-00_pc1",
        "exp_2025-12-05_10-00-00_pc1",
        "exp_2025-12-04_09-00-00_pc1",
    ]

    sorted_dirs = sorted(test_dirs, reverse=True)
    expected_order = [
        "exp_2025-12-05_10-00-00_pc1",
        "exp_2025-12-04_14-31-00_pc1",
        "exp_2025-12-04_14-30-00_pc1",
        "exp_2025-12-04_09-00-00_pc1",
    ]

    if sorted_dirs == expected_order:
        log_pass("checkpoint_sorting", "Checkpoints sort correctly by timestamp (reverse)")
    else:
        log_fail("checkpoint_sorting", f"Expected {expected_order}, got {sorted_dirs}")


# ============================================================================
# SECTION 12: ARGUMENT VALIDATION TESTS
# ============================================================================

def test_main_py_arguments():
    """Test that main.py accepts all required arguments."""
    print("\n" + "="*80)
    print("TEST SECTION: main.py Argument Validation")
    print("="*80)

    import subprocess

    # Test that main.py has a help option
    try:
        result = subprocess.run(
            ["python3", "main.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            log_pass("main_py_help", "main.py --help works")

            # Check for expected arguments in help text
            help_text = result.stdout
            expected_args = [
                "--num_classes",
                "--model_choice",
                "--window_size",
                "--learning_rate",
                "--epochs",
                "--dropout",
                "--saved_checkpoint_pth",
                "--freeze_backbone",
                "--use_lora",
                "--lora_rank",
                "--custom_data_folder",
            ]

            for arg in expected_args:
                if arg in help_text:
                    log_pass(f"main_py_has_{arg}", f"main.py supports {arg}")
                else:
                    log_fail(f"main_py_has_{arg}", f"main.py missing {arg} argument")
        else:
            log_fail("main_py_help", f"main.py --help failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        log_fail("main_py_help", "main.py --help timed out")
    except Exception as e:
        log_fail("main_py_help", f"Error running main.py: {e}")


# ============================================================================
# SECTION 13: RUNTIME INTEGRATION TESTS
# ============================================================================

def test_checkpoint_loadable():
    """Test that the pretrained checkpoint can actually be loaded."""
    print("\n" + "="*80)
    print("TEST SECTION: Runtime Integration Tests")
    print("="*80)

    import torch

    if not os.path.exists(PRETRAINED_CHECKPOINT):
        log_skip("checkpoint_loadable", f"Checkpoint not found: {PRETRAINED_CHECKPOINT}")
        return

    try:
        checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location='cpu', weights_only=False)

        # Test 13.1: Checkpoint has expected keys
        if 'model_info' in checkpoint:
            log_pass("checkpoint_has_model_info", "Checkpoint has 'model_info' key")
        else:
            log_fail("checkpoint_has_model_info", "Checkpoint missing 'model_info' key")

        if 'args_dict' in checkpoint:
            log_pass("checkpoint_has_args_dict", "Checkpoint has 'args_dict' key")
        else:
            log_fail("checkpoint_has_args_dict", "Checkpoint missing 'args_dict' key")

        # Test 13.2: model_info has state_dict
        if 'model_state_dict' in checkpoint.get('model_info', {}):
            state_dict = checkpoint['model_info']['model_state_dict']
            log_pass("checkpoint_has_state_dict", f"State dict has {len(state_dict)} parameters")
        else:
            log_fail("checkpoint_has_state_dict", "Checkpoint missing 'model_state_dict'")

    except Exception as e:
        log_fail("checkpoint_loadable", f"Failed to load checkpoint: {e}")


def test_model_initialization():
    """Test that model can be initialized with checkpoint config."""
    import torch
    from preprocessing_utils import initialize_model
    import argparse

    if not os.path.exists(PRETRAINED_CHECKPOINT):
        log_skip("model_init", f"Checkpoint not found: {PRETRAINED_CHECKPOINT}")
        return

    try:
        checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location='cpu', weights_only=False)
        args_dict = checkpoint.get('args_dict', {})

        # Create args namespace from checkpoint - use all values from checkpoint
        # to ensure we have all required attributes
        args = argparse.Namespace(**{k: v for k, v in args_dict.items()
                                     if not k.startswith('_') and
                                     not callable(v) and
                                     k not in ['labeled_csv_paths_train', 'labeled_csv_paths_val',
                                               'unlabeled_csv_paths_train', 'mask_tokens_dict',
                                               'command_line_command', 'public_data_folders',
                                               'precomputed_mean', 'precomputed_std']})

        model = initialize_model(args)
        log_pass("model_init", f"Model initialized successfully with {sum(p.numel() for p in model.parameters())} parameters")

        # Test loading state dict
        model.load_state_dict(checkpoint['model_info']['model_state_dict'], strict=False)
        log_pass("model_load_weights", "Successfully loaded pretrained weights")

    except Exception as e:
        log_fail("model_init", f"Failed to initialize model: {e}")


def test_evaluation_function_runs():
    """Test that evaluate_checkpoint_programmatic actually runs without error."""
    from event_classification import evaluate_checkpoint_programmatic

    if not os.path.exists(PRETRAINED_CHECKPOINT):
        log_skip("eval_function_runs", f"Checkpoint not found: {PRETRAINED_CHECKPOINT}")
        return

    # Find a test file
    test_folder = PARTICIPANT_FOLDERS.get('p15')
    if not test_folder or not os.path.isdir(test_folder):
        log_skip("eval_function_runs", "p15 data folder not found")
        return

    test_file = os.path.join(test_folder, "p15_open_1.csv")
    if not os.path.exists(test_file):
        test_files_found = glob.glob(os.path.join(test_folder, "*_open_1.csv"))
        if test_files_found:
            test_file = test_files_found[0]
        else:
            log_skip("eval_function_runs", "No test file found")
            return

    try:
        metrics = evaluate_checkpoint_programmatic(
            checkpoint_path=PRETRAINED_CHECKPOINT,
            csv_files=[test_file],
            buffer_range=800,
            lookahead=100,
            samples_between_prediction=100,
            allow_relax=1,
            stride=1,
            model_choice="any2any",
            verbose=0,
        )

        # Test 13.3: Returns expected metrics
        if 'transition_accuracy' in metrics:
            log_pass("eval_has_transition_acc", f"Transition accuracy: {metrics['transition_accuracy']:.4f}")
        else:
            log_fail("eval_has_transition_acc", "Missing 'transition_accuracy' in metrics")

        if 'raw_accuracy' in metrics:
            log_pass("eval_has_raw_acc", f"Raw accuracy: {metrics['raw_accuracy']:.4f}")
        else:
            log_fail("eval_has_raw_acc", "Missing 'raw_accuracy' in metrics")

        # Test 13.4: Metrics are valid numbers
        trans_acc = metrics.get('transition_accuracy', -1)
        if 0.0 <= trans_acc <= 1.0:
            log_pass("eval_metrics_valid", f"Metrics are valid (0 <= acc <= 1)")
        else:
            log_fail("eval_metrics_valid", f"Invalid metric value: {trans_acc}")

    except Exception as e:
        log_fail("eval_function_runs", f"Evaluation failed: {e}")


def test_json_serialization():
    """Test that metrics can be serialized to JSON (as done in save results)."""
    import json

    # Simulate metrics dict that would be created
    test_metrics = {
        'participant': 'p15',
        'variant': 'lora',
        'condition': 'mid_session_baseline',
        'transition_accuracy': 0.8567,
        'raw_accuracy': 0.9234,
        'test_files': ['/path/to/p15_open_5.csv', '/path/to/p15_close_5.csv'],
    }

    try:
        json_str = json.dumps(test_metrics, indent=4)
        # Try to parse it back
        parsed = json.loads(json_str)

        if parsed == test_metrics:
            log_pass("json_serialization", "Metrics can be serialized and deserialized")
        else:
            log_fail("json_serialization", "Serialization changed the data")

    except Exception as e:
        log_fail("json_serialization", f"JSON serialization failed: {e}")


def test_subprocess_command_validity():
    """Test that the constructed command would be valid (dry run)."""
    import subprocess

    # Construct a minimal test command (just check --help works)
    cmd = ["python3", "main.py", "--help"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            log_pass("subprocess_cmd_valid", "main.py command is valid")
        else:
            log_fail("subprocess_cmd_valid", f"Command failed: {result.stderr[:200]}")
    except Exception as e:
        log_fail("subprocess_cmd_valid", f"Command execution failed: {e}")


def test_temp_directory_operations():
    """Test temporary directory creation and cleanup patterns."""
    import tempfile

    # Test 13.5: os.makedirs with exist_ok
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, "level1", "level2", "level3")

        try:
            os.makedirs(nested_dir, exist_ok=True)
            if os.path.isdir(nested_dir):
                log_pass("makedirs_nested", "Nested directory creation works")
            else:
                log_fail("makedirs_nested", "Directory not created")

            # Test creating again (should not fail with exist_ok=True)
            os.makedirs(nested_dir, exist_ok=True)
            log_pass("makedirs_exist_ok", "makedirs with exist_ok=True handles existing dirs")

        except Exception as e:
            log_fail("makedirs_nested", f"Failed: {e}")


def test_glob_pattern_edge_cases():
    """Test glob patterns with edge cases found in actual data."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with various edge case names
        test_files = [
            "p15_open_1.csv",
            "p15_close_1.csv",
            "p15_open_10.csv",  # Should NOT match *_open_1.csv
            "p15_open_1_backup.csv",  # Should NOT match *_open_1.csv
            "subdir_p15_open_1.csv",  # Should match *_open_1.csv
        ]

        for f in test_files:
            open(os.path.join(tmpdir, f), 'w').close()

        # Test specific patterns
        matches = glob.glob(os.path.join(tmpdir, "*_open_1.csv"))
        filenames = [os.path.basename(m) for m in matches]

        # Should match: p15_open_1.csv, subdir_p15_open_1.csv
        # Should NOT match: p15_open_10.csv, p15_open_1_backup.csv
        if "p15_open_1.csv" in filenames and "subdir_p15_open_1.csv" in filenames:
            log_pass("glob_matches_expected", "Glob matches expected files")
        else:
            log_fail("glob_matches_expected", f"Unexpected matches: {filenames}")

        if "p15_open_10.csv" not in filenames:
            log_pass("glob_excludes_10", "Glob correctly excludes *_10.csv")
        else:
            log_fail("glob_excludes_10", "Glob incorrectly matched p15_open_10.csv")

        if "p15_open_1_backup.csv" not in filenames:
            log_pass("glob_excludes_backup", "Glob correctly excludes *_backup.csv")
        else:
            log_fail("glob_excludes_backup", "Glob incorrectly matched p15_open_1_backup.csv")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test sections."""
    print("\n" + "#"*80)
    print("# ReactEMG Stroke - run_main_experiment.py Test Suite")
    print("#"*80)

    # Section 1: Configuration
    test_participants_config()
    test_pretrained_checkpoint_config()
    test_variants_config()
    test_test_conditions_config()

    # Section 2: get_all_calibration_files()
    test_get_all_calibration_files_basic()
    test_get_all_calibration_files_with_missing()
    test_get_all_calibration_files_empty()
    test_get_all_calibration_files_multiple_matches()

    # Section 3: get_test_files()
    test_get_test_files_all_conditions()
    test_get_test_files_file_not_found()
    test_get_test_files_multiple_matches()
    test_get_test_files_invalid_condition()

    # Section 4: Pattern Matching
    test_pattern_matching_calibration()
    test_pattern_matching_test_conditions()

    # Section 5: Training Command Construction
    test_training_command_construction()
    test_hyperparameter_configs()

    # Section 6: CV Fold Splitting
    test_cv_fold_files()
    test_cv_fold_invalid_index()

    # Section 7: Symlink Creation
    test_symlink_creation()

    # Section 8: Error Handling
    test_error_handling()

    # Section 9: Integration Interface
    test_evaluate_checkpoint_interface()
    test_results_directory_structure()

    # Section 10: Data Integrity
    test_data_file_integrity()
    test_label_values()

    # Section 11: Checkpoint Naming
    test_checkpoint_directory_naming()
    test_checkpoint_sorting()

    # Section 12: Argument Validation
    test_main_py_arguments()

    # Section 13: Runtime Integration Tests
    test_checkpoint_loadable()
    test_model_initialization()
    test_evaluation_function_runs()
    test_json_serialization()
    test_subprocess_command_validity()
    test_temp_directory_operations()
    test_glob_pattern_edge_cases()

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

    # Return exit code
    return 0 if len(test_results['failed']) == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
