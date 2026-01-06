"""
Master script for running the main ReactEMG stroke experiment.

This script:
1. Runs hyperparameter search for each variant using 4-fold CV
2. Trains final models with best hyperparameters on full calibration pool
3. Evaluates all models on all test conditions
4. Organizes results in structured output directories
"""

import os
import sys
import subprocess
import json
import glob
import shutil
from typing import Dict, List
from cv_hyperparameter_search import hyperparameter_search
from event_classification import evaluate_checkpoint_programmatic


# Configuration
PARTICIPANTS = {
    'p4': '~/Workspace/myhand/src/collected_data/2026_01_06',
    'p15': '~/Workspace/myhand/src/collected_data/2025_12_04',
    'p20': '~/Workspace/myhand/src/collected_data/2025_12_18',
}

PRETRAINED_CHECKPOINT = "/home/rsw1/Workspace/reactemg/reactemg/model_checkpoints/LOSO_s14_left_2025-11-15_19-01-41_pc1/epoch_4.pth"

VARIANTS = ['stroke_only', 'head_only', 'lora', 'full_finetune']

TEST_CONDITIONS = {
    'mid_session_baseline': ['open_5.csv', 'close_5.csv'],
    'end_session_baseline': ['open_fatigue.csv', 'close_fatigue.csv'],
    'unseen_posture': ['open_hovering.csv', 'close_hovering.csv'],
    'sensor_shift': ['open_sensor_shift.csv', 'close_sensor_shift.csv'],
    'orthosis_actuated': ['close_from_open.csv'],  # Only close
}


def get_all_calibration_files(participant_folder: str) -> List[str]:
    """Get all calibration files (open_1-4, close_1-4)."""
    calib_files = []
    for set_num in range(1, 5):
        open_file = glob.glob(os.path.join(participant_folder, f"*_open_{set_num}.csv"))
        close_file = glob.glob(os.path.join(participant_folder, f"*_close_{set_num}.csv"))
        if len(open_file) == 1 and len(close_file) == 1:
            calib_files.extend([open_file[0], close_file[0]])
    return calib_files


def get_test_files(participant_folder: str, condition: str) -> List[str]:
    """Get test files for a specific condition."""
    file_patterns = TEST_CONDITIONS[condition]
    test_files = []
    for pattern in file_patterns:
        files = glob.glob(os.path.join(participant_folder, f"*_{pattern}"))
        if len(files) == 1:
            test_files.append(files[0])
        elif len(files) == 0:
            raise FileNotFoundError(
                f"No files found matching pattern '*_{pattern}' in {participant_folder}. "
                f"Expected file for condition '{condition}'."
            )
        else:
            raise ValueError(
                f"Found {len(files)} files matching pattern '*_{pattern}' in {participant_folder}, "
                f"expected exactly 1. Matches: {files}"
            )
    return test_files


def train_final_model(
    participant: str,
    participant_folder: str,
    variant: str,
    best_config: Dict,
    pretrained_checkpoint: str,
) -> str:
    """
    Train final model with best hyperparameters on full calibration pool.

    Returns:
        Path to saved checkpoint
    """
    print(f"\n{'='*80}")
    print(f"Training Final Model: {participant} - {variant}")
    print(f"Config: {best_config}")
    print(f"{'='*80}\n")

    # Get all calibration files
    calib_files = get_all_calibration_files(participant_folder)
    print(f"Training on {len(calib_files)} calibration files")

    # Create training folder
    train_dir = os.path.join("temp_final_training", f"{participant}_{variant}")
    os.makedirs(train_dir, exist_ok=True)

    # Symlink calibration files
    for calib_file in calib_files:
        link_path = os.path.join(train_dir, os.path.basename(calib_file))
        # Use lexists() to detect broken symlinks (exists() returns False for broken symlinks)
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(calib_file, link_path)

    # Build training command
    exp_name = f"{participant}_{variant}_final"

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
        "--exp_name", exp_name,
        "--custom_data_folder", train_dir,
    ]

    # Add variant-specific flags
    if variant == 'stroke_only':
        pass  # No pretrained checkpoint
    elif variant == 'head_only':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
        cmd.extend(["--freeze_backbone", "1"])
    elif variant == 'lora':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
        cmd.extend(["--use_lora", "1", "--lora_rank", "16", "--lora_alpha", "8", "--lora_dropout_p", "0.05"])
    elif variant == 'full_finetune':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Run training
    subprocess.run(cmd, check=True)

    # Find saved checkpoint
    # Sort by name descending to get the most recent (timestamps are in YYYY-MM-DD_HH-MM-SS format)
    checkpoint_dir_pattern = f"model_checkpoints/{exp_name}_*"
    checkpoint_dirs = sorted(glob.glob(checkpoint_dir_pattern), reverse=True)
    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"No checkpoint found matching pattern: {checkpoint_dir_pattern}")

    checkpoint_dir = checkpoint_dirs[0]
    if len(checkpoint_dirs) > 1:
        print(f"Note: Found {len(checkpoint_dirs)} checkpoint directories matching '{exp_name}_*', "
              f"using most recent: {os.path.basename(checkpoint_dir)}")
    # Epochs are 0-indexed, so final epoch is epochs-1
    final_epoch = best_config['epochs'] - 1
    epoch_files = glob.glob(os.path.join(checkpoint_dir, f"epoch_{final_epoch}.pth"))
    if len(epoch_files) == 0:
        raise FileNotFoundError(f"No epoch_{final_epoch}.pth found in {checkpoint_dir}")

    checkpoint_path = epoch_files[0]

    # Copy to organized location
    final_checkpoint_dir = f"model_checkpoints/main_experiment"
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(final_checkpoint_dir, f"{participant}_{variant}_final.pth")
    shutil.copy(checkpoint_path, final_checkpoint_path)

    print(f"Final checkpoint saved to: {final_checkpoint_path}")

    return final_checkpoint_path


def evaluate_all_conditions(
    participant: str,
    participant_folder: str,
    variant: str,
    checkpoint_path: str,
    results_base_dir: str = "results/main_experiment",
):
    """Evaluate model on all test conditions and save results."""

    print(f"\n{'='*80}")
    print(f"Evaluating: {participant} - {variant}")
    print(f"{'='*80}\n")

    for condition, file_patterns in TEST_CONDITIONS.items():
        print(f"\nEvaluating on: {condition}")

        test_files = get_test_files(participant_folder, condition)
        if len(test_files) == 0:
            print(f"Warning: No test files found for {condition}")
            continue

        # Evaluate (with latency computation)
        metrics = evaluate_checkpoint_programmatic(
            checkpoint_path=checkpoint_path,
            csv_files=test_files,
            buffer_range=800,
            lookahead=100,
            samples_between_prediction=100,
            allow_relax=1,
            stride=1,
            model_choice="any2any",
            verbose=1,  # Save detailed results
            compute_latency=True,  # Compute detection latency
        )

        # Save metrics summary (including latency)
        results_dir = os.path.join(results_base_dir, participant, variant, condition)
        os.makedirs(results_dir, exist_ok=True)

        metrics_file = os.path.join(results_dir, "metrics_summary.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'participant': participant,
                'variant': variant,
                'condition': condition,
                'transition_accuracy': float(metrics['transition_accuracy']),
                'raw_accuracy': float(metrics['raw_accuracy']),
                'average_latency': float(metrics['average_latency']),
                'std_latency': float(metrics['std_latency']),
                'median_latency': float(metrics['median_latency']),
                'min_latency': int(metrics['min_latency']),
                'max_latency': int(metrics['max_latency']),
                'num_latency_samples': int(metrics['num_latency_samples']),
                'test_files': test_files,
            }, f, indent=4)

        print(f"  Transition Acc: {metrics['transition_accuracy']:.4f}")
        print(f"  Raw Acc: {metrics['raw_accuracy']:.4f}")
        print(f"  Avg Latency: {metrics['average_latency']:.1f} ± {metrics['std_latency']:.1f} timesteps")
        print(f"  Results saved to: {results_dir}")


def run_zero_shot_evaluation(
    participant: str,
    participant_folder: str,
    pretrained_checkpoint: str,
):
    """Evaluate pretrained model zero-shot on all test conditions."""

    print(f"\n{'='*80}")
    print(f"Zero-Shot Evaluation: {participant}")
    print(f"{'='*80}\n")

    evaluate_all_conditions(
        participant=participant,
        participant_folder=participant_folder,
        variant='zero_shot',
        checkpoint_path=pretrained_checkpoint,
    )


def run_main_experiment(participant_filter='all'):
    """
    Run the complete main experiment.

    Args:
        participant_filter: Which participant(s) to run ('p15', 'p20', or 'all')
    """

    print("\n" + "="*80)
    print("ReactEMG Stroke - Main Experiment")
    if participant_filter != 'all':
        print(f"Running for participant: {participant_filter}")
    print("="*80 + "\n")

    # Filter participants based on argument
    if participant_filter == 'all':
        participants_to_run = PARTICIPANTS
    else:
        if participant_filter not in PARTICIPANTS:
            raise ValueError(f"Unknown participant: {participant_filter}. Choose from: {list(PARTICIPANTS.keys())}")
        participants_to_run = {participant_filter: PARTICIPANTS[participant_filter]}

    for participant, participant_folder in participants_to_run.items():
        participant_folder = os.path.expanduser(participant_folder)

        print(f"\n{'#'*80}")
        print(f"# Participant: {participant}")
        print(f"{'#'*80}\n")

        # 1. Zero-shot evaluation
        print(f"\n--- Zero-Shot Evaluation ---")
        run_zero_shot_evaluation(participant, participant_folder, PRETRAINED_CHECKPOINT)

        # 2. For each trainable variant
        for variant in VARIANTS:
            print(f"\n{'='*80}")
            print(f"Variant: {variant}")
            print(f"{'='*80}")

            # Step 1: Hyperparameter search
            print(f"\n--- Step 1: Hyperparameter Search ---")
            best_config = hyperparameter_search(
                participant=participant,
                participant_folder=participant_folder,
                variant=variant,
                pretrained_checkpoint=PRETRAINED_CHECKPOINT,
                temp_dir="temp_cv_checkpoints",
            )

            # Step 2: Train final model
            print(f"\n--- Step 2: Train Final Model ---")
            final_checkpoint = train_final_model(
                participant=participant,
                participant_folder=participant_folder,
                variant=variant,
                best_config=best_config,
                pretrained_checkpoint=PRETRAINED_CHECKPOINT,
            )

            # Step 3: Evaluate on all test conditions
            print(f"\n--- Step 3: Evaluate on Test Sets ---")
            evaluate_all_conditions(
                participant=participant,
                participant_folder=participant_folder,
                variant=variant,
                checkpoint_path=final_checkpoint,
            )

    print("\n" + "="*80)
    print("Main Experiment Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ReactEMG Stroke - Main Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run only participant p4
  python3 run_main_experiment.py --participant p4

  # Run only participant p15
  python3 run_main_experiment.py --participant p15

  # Run only participant p20
  python3 run_main_experiment.py --participant p20

  # Run all participants
  python3 run_main_experiment.py --participant all
  python3 run_main_experiment.py  # (default)
        """
    )

    parser.add_argument(
        "--participant",
        type=str,
        choices=['p4', 'p15', 'p20', 'all'],
        default='all',
        help="Which participant to run: p4, p15, p20, or all (default: all)"
    )

    args = parser.parse_args()

    run_main_experiment(participant_filter=args.participant)
