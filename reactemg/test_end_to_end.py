"""
End-to-end integration tests for ReactEMG stroke experiments.

These tests actually run training and evaluation to catch runtime errors
that unit tests cannot detect. Run these before starting multi-day experiments!

Usage:
    python3 test_end_to_end.py

This will take ~10-20 minutes but will save you days of wasted GPU time.
"""

import os
import sys
import shutil
import glob
import json
import subprocess
import tempfile
from pathlib import Path

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}


def log_pass(test_name: str, message: str = ""):
    """Log a passing test."""
    test_results['passed'].append(test_name)
    print(f"✓ {test_name}")
    if message:
        print(f"  {message}")


def log_fail(test_name: str, error: str):
    """Log a failing test."""
    test_results['failed'].append((test_name, error))
    print(f"✗ {test_name}")
    print(f"  ERROR: {error}")


def log_warning(test_name: str, warning: str):
    """Log a warning."""
    test_results['warnings'].append((test_name, warning))
    print(f"⚠ {test_name}")
    print(f"  WARNING: {warning}")


# ============================================================================
# CONFIGURATION
# ============================================================================

PARTICIPANT_FOLDER = os.path.expanduser("~/Workspace/myhand/src/collected_data/2025_12_04")
PRETRAINED_CHECKPOINT = "/home/rsw1/Workspace/reactemg/reactemg/model_checkpoints/LOSO_s14_left_2025-11-15_19-01-41_pc1/epoch_4.pth"


# ============================================================================
# END-TO-END INTEGRATION TESTS
# ============================================================================

def test_e2e_training_smoke_test():
    """
    Smoke test: Train a tiny model for 1 epoch to verify the training pipeline works.
    """
    print("\n" + "="*80)
    print("E2E TEST 1: Training Smoke Test (1 epoch, tiny model)")
    print("="*80)

    if not os.path.exists(PARTICIPANT_FOLDER):
        log_warning("test_e2e_training_smoke_test", f"Test data not found at {PARTICIPANT_FOLDER}")
        return

    try:
        # Create a temporary directory for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create symlinks to just 2 files (one open, one close) for fast testing
            temp_data_dir = os.path.join(temp_dir, "data")
            os.makedirs(temp_data_dir)

            open_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_open_1.csv"))[0]
            close_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_close_1.csv"))[0]

            os.symlink(open_file, os.path.join(temp_data_dir, os.path.basename(open_file)))
            os.symlink(close_file, os.path.join(temp_data_dir, os.path.basename(close_file)))

            # Run training with minimal configuration
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
                "--batch_size", "16",  # Small batch for speed
                "--learning_rate", "1e-4",
                "--epochs", "1",  # Just 1 epoch for smoke test
                "--dropout", "0.1",
                "--exp_name", "e2e_smoke_test",
                "--custom_data_folder", temp_data_dir,
            ]

            print("  Running training...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )

            if result.returncode != 0:
                log_fail("test_e2e_training_smoke_test",
                        f"Training failed with code {result.returncode}\nStderr: {result.stderr[-500:]}")
                return

            # Check that checkpoint was created
            checkpoint_dirs = glob.glob("model_checkpoints/e2e_smoke_test_*")
            if len(checkpoint_dirs) == 0:
                log_fail("test_e2e_training_smoke_test", "No checkpoint directory created")
                return

            checkpoint_dir = checkpoint_dirs[0]

            # Check for epoch_0.pth (0-indexed)
            epoch_file = os.path.join(checkpoint_dir, "epoch_0.pth")
            if not os.path.exists(epoch_file):
                log_fail("test_e2e_training_smoke_test", f"No epoch_0.pth found in {checkpoint_dir}")
                return

            # Clean up
            shutil.rmtree(checkpoint_dir)

            log_pass("test_e2e_training_smoke_test", "Training completed and checkpoint saved correctly")

    except subprocess.TimeoutExpired:
        log_fail("test_e2e_training_smoke_test", "Training timed out after 5 minutes")
    except Exception as e:
        log_fail("test_e2e_training_smoke_test", str(e))


def test_e2e_checkpoint_epoch_indexing():
    """
    Test that epoch indexing is correct for different epoch counts.
    """
    print("\n" + "="*80)
    print("E2E TEST 2: Checkpoint Epoch Indexing")
    print("="*80)

    if not os.path.exists(PARTICIPANT_FOLDER):
        log_warning("test_e2e_checkpoint_epoch_indexing", "Test data not found")
        return

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_data_dir = os.path.join(temp_dir, "data")
            os.makedirs(temp_data_dir)

            open_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_open_1.csv"))[0]
            close_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_close_1.csv"))[0]

            os.symlink(open_file, os.path.join(temp_data_dir, os.path.basename(open_file)))
            os.symlink(close_file, os.path.join(temp_data_dir, os.path.basename(close_file)))

            # Test with epochs=3 (should create epoch_0, epoch_1, epoch_2)
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
                "--batch_size", "16",
                "--learning_rate", "1e-4",
                "--epochs", "3",
                "--dropout", "0.1",
                "--exp_name", "e2e_epoch_test",
                "--custom_data_folder", temp_data_dir,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                log_fail("test_e2e_checkpoint_epoch_indexing", "Training failed")
                return

            checkpoint_dirs = glob.glob("model_checkpoints/e2e_epoch_test_*")
            if len(checkpoint_dirs) == 0:
                log_fail("test_e2e_checkpoint_epoch_indexing", "No checkpoint directory")
                return

            checkpoint_dir = checkpoint_dirs[0]

            # Should have epoch_0, epoch_1, epoch_2 (NOT epoch_3)
            expected_epochs = [0, 1, 2]
            for epoch in expected_epochs:
                epoch_file = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
                if not os.path.exists(epoch_file):
                    log_fail("test_e2e_checkpoint_epoch_indexing",
                            f"Missing epoch_{epoch}.pth")
                    shutil.rmtree(checkpoint_dir)
                    return

            # Should NOT have epoch_3
            epoch_3 = os.path.join(checkpoint_dir, "epoch_3.pth")
            if os.path.exists(epoch_3):
                log_fail("test_e2e_checkpoint_epoch_indexing",
                        "Unexpected epoch_3.pth found (epochs should be 0-indexed)")
                shutil.rmtree(checkpoint_dir)
                return

            shutil.rmtree(checkpoint_dir)
            log_pass("test_e2e_checkpoint_epoch_indexing",
                    "Checkpoints correctly saved as epoch_0, epoch_1, epoch_2")

    except subprocess.TimeoutExpired:
        log_fail("test_e2e_checkpoint_epoch_indexing", "Training timed out")
    except Exception as e:
        log_fail("test_e2e_checkpoint_epoch_indexing", str(e))


def test_e2e_freeze_backbone_training():
    """
    Test that freeze_backbone training actually works end-to-end.
    """
    print("\n" + "="*80)
    print("E2E TEST 3: Freeze Backbone Training")
    print("="*80)

    if not os.path.exists(PARTICIPANT_FOLDER):
        log_warning("test_e2e_freeze_backbone_training", "Test data not found")
        return

    if not os.path.exists(PRETRAINED_CHECKPOINT):
        log_warning("test_e2e_freeze_backbone_training", "Pretrained checkpoint not found")
        return

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_data_dir = os.path.join(temp_dir, "data")
            os.makedirs(temp_data_dir)

            open_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_open_1.csv"))[0]
            close_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_close_1.csv"))[0]

            os.symlink(open_file, os.path.join(temp_data_dir, os.path.basename(open_file)))
            os.symlink(close_file, os.path.join(temp_data_dir, os.path.basename(close_file)))

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
                "--batch_size", "16",
                "--learning_rate", "1e-4",
                "--epochs", "1",
                "--dropout", "0.1",
                "--exp_name", "e2e_freeze_test",
                "--custom_data_folder", temp_data_dir,
                "--freeze_backbone", "1",
                "--saved_checkpoint_pth", PRETRAINED_CHECKPOINT,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                log_fail("test_e2e_freeze_backbone_training",
                        f"Training failed: {result.stderr[-500:]}")
                return

            checkpoint_dirs = glob.glob("model_checkpoints/e2e_freeze_test_*")
            if len(checkpoint_dirs) == 0:
                log_fail("test_e2e_freeze_backbone_training", "No checkpoint created")
                return

            shutil.rmtree(checkpoint_dirs[0])
            log_pass("test_e2e_freeze_backbone_training",
                    "Freeze backbone training completed successfully")

    except subprocess.TimeoutExpired:
        log_fail("test_e2e_freeze_backbone_training", "Training timed out")
    except Exception as e:
        log_fail("test_e2e_freeze_backbone_training", str(e))


def test_e2e_lora_training():
    """
    Test that LoRA training works end-to-end.
    """
    print("\n" + "="*80)
    print("E2E TEST 4: LoRA Training")
    print("="*80)

    if not os.path.exists(PARTICIPANT_FOLDER):
        log_warning("test_e2e_lora_training", "Test data not found")
        return

    if not os.path.exists(PRETRAINED_CHECKPOINT):
        log_warning("test_e2e_lora_training", "Pretrained checkpoint not found")
        return

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_data_dir = os.path.join(temp_dir, "data")
            os.makedirs(temp_data_dir)

            open_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_open_1.csv"))[0]
            close_file = glob.glob(os.path.join(PARTICIPANT_FOLDER, "*_close_1.csv"))[0]

            os.symlink(open_file, os.path.join(temp_data_dir, os.path.basename(open_file)))
            os.symlink(close_file, os.path.join(temp_data_dir, os.path.basename(close_file)))

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
                "--batch_size", "16",
                "--learning_rate", "1e-4",
                "--epochs", "1",
                "--dropout", "0.1",
                "--exp_name", "e2e_lora_test",
                "--custom_data_folder", temp_data_dir,
                "--use_lora", "1",
                "--lora_rank", "16",
                "--lora_alpha", "8",
                "--lora_dropout_p", "0.05",
                "--saved_checkpoint_pth", PRETRAINED_CHECKPOINT,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                log_fail("test_e2e_lora_training", f"Training failed: {result.stderr[-500:]}")
                return

            checkpoint_dirs = glob.glob("model_checkpoints/e2e_lora_test_*")
            if len(checkpoint_dirs) == 0:
                log_fail("test_e2e_lora_training", "No checkpoint created")
                return

            shutil.rmtree(checkpoint_dirs[0])
            log_pass("test_e2e_lora_training", "LoRA training completed successfully")

    except subprocess.TimeoutExpired:
        log_fail("test_e2e_lora_training", "Training timed out")
    except Exception as e:
        log_fail("test_e2e_lora_training", str(e))


def test_e2e_programmatic_evaluation():
    """
    Test that programmatic evaluation works on actual checkpoint.
    """
    print("\n" + "="*80)
    print("E2E TEST 5: Programmatic Evaluation")
    print("="*80)

    if not os.path.exists(PRETRAINED_CHECKPOINT):
        log_warning("test_e2e_programmatic_evaluation", "Pretrained checkpoint not found")
        return

    if not os.path.exists(PARTICIPANT_FOLDER):
        log_warning("test_e2e_programmatic_evaluation", "Test data not found")
        return

    try:
        from event_classification import evaluate_checkpoint_programmatic

        # Test files
        test_files = []
        for pattern in ["*_open_1.csv", "*_close_1.csv"]:
            files = glob.glob(os.path.join(PARTICIPANT_FOLDER, pattern))
            if len(files) == 1:
                test_files.append(files[0])

        if len(test_files) < 2:
            log_fail("test_e2e_programmatic_evaluation", "Not enough test files found")
            return

        print("  Running evaluation (this may take 30-60 seconds)...")
        metrics = evaluate_checkpoint_programmatic(
            checkpoint_path=PRETRAINED_CHECKPOINT,
            csv_files=test_files[:2],  # Just 2 files for speed
            buffer_range=800,
            lookahead=100,
            samples_between_prediction=100,
            allow_relax=1,
            stride=1,
            model_choice="any2any",
            verbose=0,
        )

        # Check that metrics are returned correctly
        if 'transition_accuracy' not in metrics:
            log_fail("test_e2e_programmatic_evaluation", "Missing transition_accuracy in output")
            return

        if 'raw_accuracy' not in metrics:
            log_fail("test_e2e_programmatic_evaluation", "Missing raw_accuracy in output")
            return

        # Metrics should be between 0 and 1
        if not (0 <= metrics['transition_accuracy'] <= 1):
            log_fail("test_e2e_programmatic_evaluation",
                    f"Invalid transition_accuracy: {metrics['transition_accuracy']}")
            return

        log_pass("test_e2e_programmatic_evaluation",
                f"Evaluation successful - Trans Acc: {metrics['transition_accuracy']:.3f}, "
                f"Raw Acc: {metrics['raw_accuracy']:.3f}")

    except Exception as e:
        log_fail("test_e2e_programmatic_evaluation", str(e))


def test_e2e_cv_fold_training():
    """
    Test that CV fold training and evaluation works.
    """
    print("\n" + "="*80)
    print("E2E TEST 6: CV Fold Training and Evaluation")
    print("="*80)

    if not os.path.exists(PARTICIPANT_FOLDER):
        log_warning("test_e2e_cv_fold_training", "Test data not found")
        return

    try:
        from cv_hyperparameter_search import get_fold_files, run_training, evaluate_on_validation

        # Get fold 0 files
        train_files, val_files = get_fold_files(PARTICIPANT_FOLDER, fold_idx=0)

        # Test config (minimal for speed)
        test_config = {
            'learning_rate': 1e-4,
            'epochs': 1,  # Just 1 epoch for testing
            'dropout': 0.1,
            'variant': 'stroke_only'
        }

        print("  Training on fold 0...")
        checkpoint_path = run_training(
            config=test_config,
            train_files=train_files,
            participant='test_p15',
            fold_idx=0,
            pretrained_checkpoint=PRETRAINED_CHECKPOINT,
            temp_dir="temp_e2e_test",
        )

        if not os.path.exists(checkpoint_path):
            log_fail("test_e2e_cv_fold_training", f"Checkpoint not created: {checkpoint_path}")
            return

        print("  Evaluating on validation set...")
        val_acc = evaluate_on_validation(checkpoint_path, val_files)

        if not (0 <= val_acc <= 1):
            log_fail("test_e2e_cv_fold_training", f"Invalid validation accuracy: {val_acc}")
            return

        # Clean up
        checkpoint_dir = os.path.dirname(checkpoint_path)
        parent_dir = os.path.dirname(checkpoint_dir)
        if os.path.exists(parent_dir):
            shutil.rmtree(parent_dir)

        if os.path.exists("temp_e2e_test"):
            shutil.rmtree("temp_e2e_test")

        log_pass("test_e2e_cv_fold_training", f"CV training successful - Val Acc: {val_acc:.3f}")

    except Exception as e:
        log_fail("test_e2e_cv_fold_training", str(e))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_e2e_tests():
    """Run all end-to-end integration tests."""
    print("\n" + "#"*80)
    print("# ReactEMG Stroke - End-to-End Integration Tests")
    print("#"*80)
    print("\nThese tests run actual training and evaluation to catch runtime errors.")
    print("This will take ~10-20 minutes but will save you days of wasted GPU time!\n")

    # Run tests
    test_e2e_training_smoke_test()
    test_e2e_checkpoint_epoch_indexing()
    test_e2e_freeze_backbone_training()
    test_e2e_lora_training()
    test_e2e_programmatic_evaluation()
    test_e2e_cv_fold_training()

    # Summary
    print("\n" + "="*80)
    print("END-TO-END TEST SUMMARY")
    print("="*80)

    total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['warnings'])

    print(f"\nTotal tests run: {total_tests}")
    print(f"✓ Passed: {len(test_results['passed'])}")
    print(f"✗ Failed: {len(test_results['failed'])}")
    print(f"⚠ Warnings: {len(test_results['warnings'])}")

    if test_results['failed']:
        print("\n" + "!"*80)
        print("FAILED TESTS:")
        print("!"*80)
        for test_name, error in test_results['failed']:
            print(f"\n✗ {test_name}")
            print(f"  ERROR: {error}")

    if test_results['warnings']:
        print("\n" + "!"*80)
        print("WARNINGS:")
        print("!"*80)
        for test_name, warning in test_results['warnings']:
            print(f"\n⚠ {test_name}")
            print(f"  {warning}")

    print("\n" + "="*80)
    if len(test_results['failed']) == 0:
        print("✓ ALL END-TO-END TESTS PASSED!")
        print("="*80)
        print("\nYour experimental pipeline is verified and ready for multi-day runs!")
        print()
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the failing tests before running large experiments.")
        print()
        return 1


if __name__ == "__main__":
    exit_code = run_all_e2e_tests()
    sys.exit(exit_code)
