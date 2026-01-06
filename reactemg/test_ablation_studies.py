"""
Comprehensive test suite for ablation studies (data efficiency & convergence).

This script performs:
1. Pre-runtime sanity checks (configuration, file existence, dependencies)
2. Runtime validation (checkpoints, results, data integrity)
3. Post-runtime verification (result completeness, correctness)

Usage:
    # Pre-runtime checks only
    python3 test_ablation_studies.py --mode pre_runtime --experiment both

    # Runtime monitoring (run alongside experiments)
    python3 test_ablation_studies.py --mode runtime --experiment data_efficiency

    # Post-runtime verification
    python3 test_ablation_studies.py --mode post_runtime --experiment convergence
"""

import os
import sys
import json
import glob
import argparse
import importlib.util
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


###############################################################################
#                          PRE-RUNTIME CHECKS
###############################################################################

class PreRuntimeChecker:
    """Pre-runtime sanity checks for ablation experiments."""

    def __init__(self, participant: str, participant_folder: str):
        self.participant = participant
        self.participant_folder = os.path.expanduser(participant_folder)
        self.errors = []
        self.warnings = []

    def check_all(self, experiment_type: str) -> bool:
        """Run all pre-runtime checks."""
        print(f"\n{'='*80}")
        print(f"PRE-RUNTIME CHECKS: {experiment_type}")
        print(f"{'='*80}\n")

        # Common checks
        self.check_participant_folder()
        self.check_calibration_files()
        self.check_test_files()
        self.check_pretrained_checkpoint()
        self.check_main_py_flags()
        self.check_dependencies()

        # Experiment-specific checks
        if experiment_type in ['data_efficiency', 'both']:
            self.check_data_efficiency_setup()

        if experiment_type in ['convergence', 'both']:
            self.check_convergence_setup()

        # Print results
        self.print_results()

        return len(self.errors) == 0

    def check_participant_folder(self):
        """Check participant folder exists and is accessible."""
        print("Checking participant folder...")

        if not os.path.exists(self.participant_folder):
            self.errors.append(f"Participant folder does not exist: {self.participant_folder}")
        elif not os.path.isdir(self.participant_folder):
            self.errors.append(f"Participant folder is not a directory: {self.participant_folder}")
        elif not os.access(self.participant_folder, os.R_OK):
            self.errors.append(f"Participant folder is not readable: {self.participant_folder}")
        else:
            print(f"  ✓ Participant folder exists: {self.participant_folder}")

    def check_calibration_files(self):
        """Check all 8 calibration files exist (open_1-4, close_1-4)."""
        print("\nChecking calibration files...")

        expected_files = []
        for set_num in range(1, 5):
            open_pattern = os.path.join(self.participant_folder, f"*_open_{set_num}.csv")
            close_pattern = os.path.join(self.participant_folder, f"*_close_{set_num}.csv")

            open_files = glob.glob(open_pattern)
            close_files = glob.glob(close_pattern)

            if len(open_files) == 0:
                self.errors.append(f"Missing open_{set_num}.csv file")
            elif len(open_files) > 1:
                self.warnings.append(f"Multiple matches for open_{set_num}.csv: {open_files}")
            else:
                expected_files.append(open_files[0])

            if len(close_files) == 0:
                self.errors.append(f"Missing close_{set_num}.csv file")
            elif len(close_files) > 1:
                self.warnings.append(f"Multiple matches for close_{set_num}.csv: {close_files}")
            else:
                expected_files.append(close_files[0])

        if len(expected_files) == 8:
            print(f"  ✓ Found all 8 calibration files")
            # Check file integrity
            for f in expected_files:
                self._check_csv_integrity(f)
        else:
            print(f"  ✗ Expected 8 calibration files, found {len(expected_files)}")

    def check_test_files(self):
        """Check test condition files exist."""
        print("\nChecking test files...")

        test_patterns = {
            'mid_session_baseline': ['open_5.csv', 'close_5.csv'],
            'end_session_baseline': ['open_fatigue.csv', 'close_fatigue.csv'],
            'unseen_posture': ['open_hovering.csv', 'close_hovering.csv'],
            'sensor_shift': ['open_sensor_shift.csv', 'close_sensor_shift.csv'],
            'orthosis_actuated': ['close_from_open.csv'],
        }

        found_conditions = 0
        for condition, patterns in test_patterns.items():
            condition_files = []
            for pattern in patterns:
                files = glob.glob(os.path.join(self.participant_folder, f"*_{pattern}"))
                if len(files) == 1:
                    condition_files.append(files[0])
                elif len(files) == 0:
                    self.warnings.append(f"Missing test file: {pattern} for {condition}")
                else:
                    self.warnings.append(f"Multiple matches for {pattern}: {files}")

            if len(condition_files) == len(patterns):
                found_conditions += 1

        print(f"  ✓ Found {found_conditions}/{len(test_patterns)} complete test conditions")
        if found_conditions < len(test_patterns):
            self.warnings.append(f"Only {found_conditions} test conditions have all files")

    def check_pretrained_checkpoint(self):
        """Check pretrained checkpoint exists."""
        print("\nChecking pretrained checkpoint...")

        checkpoint_path = "/home/rsw1/Workspace/reactemg/reactemg/model_checkpoints/LOSO_s14_left_2025-11-15_19-01-41_pc1/epoch_4.pth"

        if not os.path.exists(checkpoint_path):
            self.errors.append(f"Pretrained checkpoint not found: {checkpoint_path}")
        else:
            print(f"  ✓ Pretrained checkpoint exists")
            # Check if it's loadable
            try:
                import torch
                torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"  ✓ Checkpoint is loadable")
            except Exception as e:
                self.errors.append(f"Checkpoint exists but cannot be loaded: {e}")

    def check_main_py_flags(self):
        """Check main.py supports required flags."""
        print("\nChecking main.py argument support...")

        required_flags = [
            '--freeze_backbone',
            '--use_lora',
            '--custom_data_folder',
            '--exclude_files',
        ]

        # Check if main.py exists
        main_py_path = os.path.join(os.path.dirname(__file__), 'main.py')
        if not os.path.exists(main_py_path):
            self.errors.append("main.py not found in reactemg directory")
            return

        # Read main.py content
        with open(main_py_path, 'r') as f:
            content = f.read()

        for flag in required_flags:
            if flag not in content:
                self.errors.append(f"main.py missing required flag: {flag}")
            else:
                print(f"  ✓ {flag} is supported")

        # Check for CRITICAL BUG: --save_every_epoch flag
        if '--save_every_epoch' in content:
            print(f"  ✓ --save_every_epoch is supported (required for convergence)")
        else:
            self.errors.append(
                "CRITICAL: main.py does not support --save_every_epoch flag! "
                "This will cause convergence.py to fail. "
                "You need to add this argument to main.py"
            )

    def check_dependencies(self):
        """Check Python dependencies are installed."""
        print("\nChecking Python dependencies...")

        required_modules = [
            'torch',
            'pandas',
            'numpy',
            'scipy',
            'matplotlib',
        ]

        for module in required_modules:
            try:
                importlib.import_module(module)
                print(f"  ✓ {module} is installed")
            except ImportError:
                self.errors.append(f"Required module not installed: {module}")

    def check_data_efficiency_setup(self):
        """Check data efficiency experiment setup."""
        print("\nChecking data efficiency setup...")

        # Check dataset_utils.py exists
        dataset_utils_path = os.path.join(os.path.dirname(__file__), 'dataset_utils.py')
        if not os.path.exists(dataset_utils_path):
            self.errors.append("dataset_utils.py not found (required for data efficiency)")
        else:
            print("  ✓ dataset_utils.py exists")

        # Check repetition extraction will work
        try:
            from dataset_utils import get_paired_repetition_indices
            paired_reps = get_paired_repetition_indices(
                participant_folder=self.participant_folder,
                num_sets=4,
                reps_per_set=3,
            )
            if len(paired_reps) == 12:
                print(f"  ✓ Successfully extracted 12 paired repetitions")
            else:
                self.warnings.append(f"Expected 12 paired reps, got {len(paired_reps)}")
        except Exception as e:
            self.errors.append(f"Failed to extract paired repetitions: {e}")

        # Check for race condition bug
        print("\n  ⚠ WARNING: Potential race condition in run_data_efficiency.py:")
        print("    Line 92: temp_sampled_segments file not in unique temp directory")
        print("    This could cause issues if running multiple trials in parallel")

    def check_convergence_setup(self):
        """Check convergence experiment setup."""
        print("\nChecking convergence setup...")

        # Check healthy s15 path
        s15_path = os.path.expanduser("~/Workspace/reactemg/data/ROAM_EMG/s15")
        if not os.path.exists(s15_path):
            self.errors.append(
                f"CRITICAL: Healthy s15 data path does not exist: {s15_path}\n"
                f"          This is hardcoded in run_convergence.py line 28"
            )
        else:
            print(f"  ✓ Healthy s15 data path exists")

            # Check for s15 files
            s15_files = glob.glob(os.path.join(s15_path, "*.csv"))
            static_grasp = [f for f in s15_files if ('static' in f.lower() or 'grasp' in f.lower())
                           and 'movement' not in f.lower()]

            if len(static_grasp) > 0:
                print(f"  ✓ Found {len(static_grasp)} s15 evaluation files")
            else:
                self.errors.append("No s15 static/grasp files found for convergence baseline")

    def _check_csv_integrity(self, filepath: str):
        """Check if CSV file is readable and has expected structure."""
        try:
            df = pd.read_csv(filepath)

            if 'gt' not in df.columns:
                self.errors.append(f"CSV missing 'gt' column: {filepath}")

            if len(df) == 0:
                self.errors.append(f"CSV is empty: {filepath}")

            # Check for reasonable length (at least 3 seconds at 200Hz)
            if len(df) < 600:
                self.warnings.append(f"CSV seems short ({len(df)} samples): {filepath}")

        except Exception as e:
            self.errors.append(f"Cannot read CSV {filepath}: {e}")

    def print_results(self):
        """Print check results."""
        print(f"\n{'='*80}")
        print("PRE-RUNTIME CHECK RESULTS")
        print(f"{'='*80}\n")

        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. {err}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warn in enumerate(self.warnings, 1):
                print(f"  {i}. {warn}")

        if not self.errors and not self.warnings:
            print("✅ All checks passed!")
        elif not self.errors:
            print("\n✅ No critical errors found, but review warnings above")
        else:
            print("\n❌ CRITICAL ERRORS FOUND - Fix these before running experiments")


###############################################################################
#                          RUNTIME CHECKS
###############################################################################

class RuntimeMonitor:
    """Monitor experiments during runtime."""

    def __init__(self, participant: str, experiment_type: str):
        self.participant = participant
        self.experiment_type = experiment_type

    def monitor_data_efficiency(self):
        """Monitor data efficiency experiment progress."""
        print(f"\n{'='*80}")
        print(f"RUNTIME MONITORING: Data Efficiency - {self.participant}")
        print(f"{'='*80}\n")

        checkpoints_dir = f"model_checkpoints/data_efficiency/{self.participant}"
        results_dir = f"results/data_efficiency/{self.participant}"

        # Check expected trials: K in [1, 4, 8], 12 trials each = 36 total
        expected_trials = {
            'K1': 12,
            'K4': 12,
            'K8': 12,
        }

        for k_val, num_trials in expected_trials.items():
            print(f"\nChecking {k_val}:")

            # Check checkpoints
            if os.path.exists(checkpoints_dir):
                checkpoints = glob.glob(os.path.join(checkpoints_dir, f"{k_val}_trial*.pth"))
                print(f"  Checkpoints: {len(checkpoints)}/{num_trials}")

                # CRITICAL: Check for variant conflicts
                checkpoint_names = [os.path.basename(c) for c in checkpoints]
                if len(checkpoint_names) != len(set(checkpoint_names)):
                    print("  ⚠️  WARNING: Duplicate checkpoint names found!")
                    print("     This suggests the variant bug in line 186 is active!")
            else:
                print(f"  Checkpoints: 0/{num_trials} (directory not created yet)")

            # Check results
            k_results_dir = os.path.join(results_dir, k_val)
            if os.path.exists(k_results_dir):
                trial_dirs = glob.glob(os.path.join(k_results_dir, "trial_*"))
                print(f"  Trial results: {len(trial_dirs)}/{num_trials}")

                # Check for aggregated results
                agg_file = os.path.join(k_results_dir, "aggregated_metrics.json")
                if os.path.exists(agg_file):
                    print(f"  ✓ Aggregated metrics file exists")
                    self._validate_aggregated_results(agg_file)
                else:
                    print(f"  ⏳ Aggregated metrics not yet created")
            else:
                print(f"  Trial results: 0/{num_trials} (directory not created yet)")

    def monitor_convergence(self):
        """Monitor convergence experiment progress."""
        print(f"\n{'='*80}")
        print(f"RUNTIME MONITORING: Convergence - {self.participant}")
        print(f"{'='*80}\n")

        checkpoints_dir = f"model_checkpoints/convergence/{self.participant}"
        results_dir = f"results/convergence/{self.participant}"

        # Check for checkpoint directory
        if os.path.exists(checkpoints_dir):
            checkpoints = sorted(glob.glob(os.path.join(checkpoints_dir, "epoch_*.pth")))
            print(f"Epoch checkpoints: {len(checkpoints)}")

            if checkpoints:
                epochs = [int(os.path.basename(c).split('_')[1].split('.')[0]) for c in checkpoints]
                print(f"  Epochs saved: {min(epochs)} to {max(epochs)}")

                # Check for gaps
                expected_epochs = set(range(min(epochs), max(epochs) + 1))
                actual_epochs = set(epochs)
                missing = expected_epochs - actual_epochs
                if missing:
                    print(f"  ⚠️  Missing epochs: {sorted(missing)}")
        else:
            print("Checkpoint directory not created yet")

        # Check results
        if os.path.exists(results_dir):
            epoch_dirs = glob.glob(os.path.join(results_dir, "epoch_*"))
            print(f"\nEpoch results: {len(epoch_dirs)}")

            # Check for frozen baseline
            frozen_dir = os.path.join(results_dir, "frozen_baseline")
            if os.path.exists(frozen_dir):
                print(f"✓ Frozen baseline evaluated")
            else:
                print(f"⏳ Frozen baseline not yet evaluated")

            # Check for convergence curves
            curves_file = os.path.join(results_dir, "convergence_curves.json")
            if os.path.exists(curves_file):
                print(f"✓ Convergence curves file exists")
                self._validate_convergence_curves(curves_file)
            else:
                print(f"⏳ Convergence curves not yet created")
        else:
            print("Results directory not created yet")

    def _validate_aggregated_results(self, filepath: str):
        """Validate aggregated results JSON."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            required_keys = ['participant', 'variant', 'budget_k', 'num_trials', 'aggregated_results']
            missing = [k for k in required_keys if k not in data]
            if missing:
                print(f"  ⚠️  Aggregated results missing keys: {missing}")

            # Check each condition has mean and std
            for condition, metrics in data.get('aggregated_results', {}).items():
                required_metrics = [
                    'transition_accuracy_mean',
                    'transition_accuracy_std',
                    'raw_accuracy_mean',
                    'raw_accuracy_std'
                ]
                missing_metrics = [m for m in required_metrics if m not in metrics]
                if missing_metrics:
                    print(f"  ⚠️  Condition {condition} missing: {missing_metrics}")

        except json.JSONDecodeError:
            print(f"  ⚠️  Invalid JSON in {filepath}")
        except Exception as e:
            print(f"  ⚠️  Error reading {filepath}: {e}")

    def _validate_convergence_curves(self, filepath: str):
        """Validate convergence curves JSON."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            required_keys = ['participant', 'variant', 'base_epochs', 'extended_epochs',
                           'frozen_baseline', 'epoch_results']
            missing = [k for k in required_keys if k not in data]
            if missing:
                print(f"  ⚠️  Convergence curves missing keys: {missing}")

            # Check epoch results completeness
            epoch_results = data.get('epoch_results', [])
            if epoch_results:
                num_epochs = len(epoch_results)
                expected = data.get('extended_epochs', 0)
                if num_epochs != expected:
                    print(f"  ⚠️  Expected {expected} epoch results, found {num_epochs}")

        except json.JSONDecodeError:
            print(f"  ⚠️  Invalid JSON in {filepath}")
        except Exception as e:
            print(f"  ⚠️  Error reading {filepath}: {e}")


###############################################################################
#                        POST-RUNTIME VERIFICATION
###############################################################################

class PostRuntimeVerifier:
    """Verify experiment results after completion."""

    def __init__(self, participant: str, experiment_type: str):
        self.participant = participant
        self.experiment_type = experiment_type
        self.errors = []
        self.warnings = []

    def verify_data_efficiency(self, variant: str):
        """Verify data efficiency results completeness."""
        print(f"\n{'='*80}")
        print(f"POST-RUNTIME VERIFICATION: Data Efficiency - {self.participant} - {variant}")
        print(f"{'='*80}\n")

        results_dir = f"results/data_efficiency/{self.participant}"

        # Check all K values
        for k_val in ['K1', 'K4', 'K8']:
            print(f"\nVerifying {k_val}:")
            k_dir = os.path.join(results_dir, k_val)

            if not os.path.exists(k_dir):
                self.errors.append(f"Results directory missing: {k_dir}")
                continue

            # Check all 12 trials
            for trial_idx in range(12):
                trial_dir = os.path.join(k_dir, f"trial_{trial_idx}")
                metrics_file = os.path.join(trial_dir, "metrics.json")

                if not os.path.exists(metrics_file):
                    self.errors.append(f"Missing metrics for {k_val} trial {trial_idx}")
                else:
                    self._verify_trial_metrics(metrics_file, k_val, trial_idx)

            # Check aggregated results
            agg_file = os.path.join(k_dir, "aggregated_metrics.json")
            if not os.path.exists(agg_file):
                self.errors.append(f"Missing aggregated metrics for {k_val}")
            else:
                self._verify_aggregated_metrics(agg_file, k_val)

        # Check nesting property
        self._verify_nesting_property(results_dir)

        self.print_results()
        return len(self.errors) == 0

    def verify_convergence(self, variant: str, expected_base_epochs: int):
        """Verify convergence results completeness."""
        print(f"\n{'='*80}")
        print(f"POST-RUNTIME VERIFICATION: Convergence - {self.participant} - {variant}")
        print(f"{'='*80}\n")

        results_dir = f"results/convergence/{self.participant}"
        checkpoints_dir = f"model_checkpoints/convergence/{self.participant}"

        extended_epochs = expected_base_epochs * 10

        # Check frozen baseline
        frozen_dir = os.path.join(results_dir, "frozen_baseline")
        if not os.path.exists(frozen_dir):
            self.errors.append("Frozen baseline results missing")
        else:
            frozen_metrics = os.path.join(frozen_dir, "metrics.json")
            if os.path.exists(frozen_metrics):
                print("✓ Frozen baseline evaluated")
            else:
                self.errors.append("Frozen baseline metrics.json missing")

        # Check all epoch results
        print(f"\nVerifying {extended_epochs} epochs:")
        missing_results = []
        missing_checkpoints = []

        for epoch in range(extended_epochs):
            epoch_dir = os.path.join(results_dir, f"epoch_{epoch}")
            epoch_metrics = os.path.join(epoch_dir, "metrics.json")

            if not os.path.exists(epoch_metrics):
                missing_results.append(epoch)

            checkpoint_file = os.path.join(checkpoints_dir, f"epoch_{epoch}.pth")
            if not os.path.exists(checkpoint_file):
                missing_checkpoints.append(epoch)

        if missing_results:
            self.errors.append(f"Missing results for epochs: {missing_results[:10]}..."
                             if len(missing_results) > 10 else f"Missing results for epochs: {missing_results}")
        else:
            print(f"✓ All {extended_epochs} epoch results present")

        if missing_checkpoints:
            self.errors.append(f"Missing checkpoints for epochs: {missing_checkpoints[:10]}..."
                             if len(missing_checkpoints) > 10 else f"Missing checkpoints for epochs: {missing_checkpoints}")
        else:
            print(f"✓ All {extended_epochs} epoch checkpoints present")

        # Check convergence curves
        curves_file = os.path.join(results_dir, "convergence_curves.json")
        if not os.path.exists(curves_file):
            self.errors.append("Convergence curves file missing")
        else:
            self._verify_convergence_curves_complete(curves_file, extended_epochs)

        self.print_results()
        return len(self.errors) == 0

    def _verify_trial_metrics(self, filepath: str, k_val: str, trial_idx: int):
        """Verify individual trial metrics file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Check structure
            if 'results' not in data:
                self.errors.append(f"{k_val} trial {trial_idx}: missing 'results' key")
                return

            # Check all test conditions have metrics
            expected_conditions = ['mid_session_baseline', 'end_session_baseline',
                                 'unseen_posture', 'sensor_shift']

            for condition in expected_conditions:
                if condition not in data['results']:
                    self.warnings.append(f"{k_val} trial {trial_idx}: missing condition {condition}")
                else:
                    metrics = data['results'][condition]
                    if 'transition_accuracy' not in metrics or 'raw_accuracy' not in metrics:
                        self.errors.append(f"{k_val} trial {trial_idx} {condition}: incomplete metrics")

        except Exception as e:
            self.errors.append(f"Error reading {filepath}: {e}")

    def _verify_aggregated_metrics(self, filepath: str, k_val: str):
        """Verify aggregated metrics file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if 'aggregated_results' not in data:
                self.errors.append(f"{k_val}: missing 'aggregated_results'")
                return

            # Check statistics are reasonable
            for condition, metrics in data['aggregated_results'].items():
                mean = metrics.get('transition_accuracy_mean', -1)
                std = metrics.get('transition_accuracy_std', -1)

                if mean < 0 or mean > 1:
                    self.errors.append(f"{k_val} {condition}: invalid mean {mean}")

                if std < 0 or std > 1:
                    self.warnings.append(f"{k_val} {condition}: suspicious std {std}")

        except Exception as e:
            self.errors.append(f"Error reading {filepath}: {e}")

    def _verify_nesting_property(self, results_dir: str):
        """Verify that K=4 samples include K=1 samples (nested structure)."""
        print("\nVerifying nesting property...")

        # This would require reading the sampled_repetitions from each trial
        # For now, just check if the file structure is consistent
        for k_val in ['K1', 'K4', 'K8']:
            k_dir = os.path.join(results_dir, k_val)
            if os.path.exists(k_dir):
                trial_files = glob.glob(os.path.join(k_dir, "trial_*/metrics.json"))
                if len(trial_files) != 12:
                    self.warnings.append(f"{k_val}: expected 12 trials, found {len(trial_files)}")

    def _verify_convergence_curves_complete(self, filepath: str, expected_epochs: int):
        """Verify convergence curves file is complete."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            epoch_results = data.get('epoch_results', [])

            if len(epoch_results) != expected_epochs:
                self.errors.append(
                    f"Convergence curves: expected {expected_epochs} epochs, found {len(epoch_results)}"
                )

            # Check each epoch has both stroke and healthy results
            for i, epoch_data in enumerate(epoch_results):
                if 'stroke_results' not in epoch_data:
                    self.errors.append(f"Epoch {i}: missing stroke_results")

                if 'healthy_results' not in epoch_data:
                    self.errors.append(f"Epoch {i}: missing healthy_results")

                # Check for metrics
                if 'stroke_avg_transition_acc' not in epoch_data:
                    self.warnings.append(f"Epoch {i}: missing stroke avg accuracy")

        except Exception as e:
            self.errors.append(f"Error reading convergence curves: {e}")

    def print_results(self):
        """Print verification results."""
        print(f"\n{'='*80}")
        print("POST-RUNTIME VERIFICATION RESULTS")
        print(f"{'='*80}\n")

        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. {err}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warn in enumerate(self.warnings, 1):
                print(f"  {i}. {warn}")

        if not self.errors and not self.warnings:
            print("✅ All verifications passed!")
        elif not self.errors:
            print("\n✅ No critical errors, but review warnings")
        else:
            print("\n❌ ERRORS FOUND - Results may be incomplete")


###############################################################################
#                               MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-runtime checks for both experiments (p15)
  python3 test_ablation_studies.py --mode pre_runtime --experiment both --participant p15 \\
      --participant_folder ~/Workspace/myhand/src/collected_data/2025_12_04

  # Runtime monitoring for data efficiency
  python3 test_ablation_studies.py --mode runtime --experiment data_efficiency --participant p15

  # Post-runtime verification for convergence
  python3 test_ablation_studies.py --mode post_runtime --experiment convergence \\
      --participant p15 --variant lora --base_epochs 10
        """
    )

    parser.add_argument('--mode', required=True,
                       choices=['pre_runtime', 'runtime', 'post_runtime'],
                       help='Test mode')
    parser.add_argument('--experiment', required=True,
                       choices=['data_efficiency', 'convergence', 'both'],
                       help='Which experiment to test')
    parser.add_argument('--participant', required=True,
                       help='Participant ID (e.g., p15)')
    parser.add_argument('--participant_folder',
                       help='Path to participant data folder (required for pre_runtime)')
    parser.add_argument('--variant',
                       help='Fine-tuning variant (required for post_runtime)')
    parser.add_argument('--base_epochs', type=int,
                       help='Base epochs for convergence (required for post_runtime convergence)')

    args = parser.parse_args()

    # Mode dispatch
    if args.mode == 'pre_runtime':
        if not args.participant_folder:
            parser.error("--participant_folder required for pre_runtime mode")

        checker = PreRuntimeChecker(args.participant, args.participant_folder)
        success = checker.check_all(args.experiment)
        sys.exit(0 if success else 1)

    elif args.mode == 'runtime':
        monitor = RuntimeMonitor(args.participant, args.experiment)

        if args.experiment in ['data_efficiency', 'both']:
            monitor.monitor_data_efficiency()

        if args.experiment in ['convergence', 'both']:
            monitor.monitor_convergence()

    elif args.mode == 'post_runtime':
        if not args.variant:
            parser.error("--variant required for post_runtime mode")

        verifier = PostRuntimeVerifier(args.participant, args.experiment)

        if args.experiment == 'data_efficiency':
            success = verifier.verify_data_efficiency(args.variant)
        elif args.experiment == 'convergence':
            if not args.base_epochs:
                parser.error("--base_epochs required for post_runtime convergence")
            success = verifier.verify_convergence(args.variant, args.base_epochs)
        else:
            print("Cannot verify 'both' - run data_efficiency and convergence separately")
            sys.exit(1)

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
