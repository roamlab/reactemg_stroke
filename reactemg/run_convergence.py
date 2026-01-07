"""
Convergence study for ReactEMG stroke.

This script:
1. Trains a model for 10× the optimal number of epochs
2. Saves checkpoints at every epoch
3. Evaluates each checkpoint on both stroke test sets and healthy s25 data
4. Tracks convergence and potential catastrophic forgetting
"""

import os
import sys
import json
import subprocess
import glob
import numpy as np
from typing import Dict, List
from event_classification import evaluate_checkpoint_programmatic


# Configuration
PARTICIPANTS = {
    'p4': '~/Workspace/myhand/src/collected_data/2026_01_06',
    'p15': '~/Workspace/myhand/src/collected_data/2025_12_04',
    'p20': '~/Workspace/myhand/src/collected_data/2025_12_18',
}

PRETRAINED_CHECKPOINT = "/home/rsw1/Workspace/reactemg/reactemg/model_checkpoints/LOSO_s14_left_2025-11-15_19-01-41_pc1/epoch_4.pth"

# HEALTHY_S25_PATH configured via command line argument (see --healthy_s25_path below)

TEST_CONDITIONS = {
    'mid_session_baseline': ['open_5.csv', 'close_5.csv'],
    'end_session_baseline': ['open_fatigue.csv', 'close_fatigue.csv'],
    'unseen_posture': ['open_hovering.csv', 'close_hovering.csv'],
    'sensor_shift': ['open_sensor_shift.csv', 'close_sensor_shift.csv'],
    'orthosis_actuated': ['close_from_open.csv'],
}


def get_healthy_s25_files(s25_path: str = None) -> List[str]:
    """Get healthy s25 evaluation files (static + grasp, exclude movement)."""
    if s25_path is None:
        s25_path = os.path.expanduser("~/Workspace/reactemg/data/ROAM_EMG/s25")
    else:
        s25_path = os.path.expanduser(s25_path)

    # Validate path exists
    if not os.path.exists(s25_path):
        raise FileNotFoundError(
            f"Healthy s25 path does not exist: {s25_path}\n"
            f"Please provide correct path with --healthy_s25_path argument"
        )

    all_files = glob.glob(os.path.join(s25_path, "*.csv"))

    # Filter for static and grasp files
    s25_files = [
        f for f in all_files
        if ('_static_' in f or '_grasp_' in f) and 'movement' not in f.lower()
    ]

    if len(s25_files) == 0:
        raise ValueError(
            f"No s25 files found matching filter (static/grasp, excluding movement) in {s25_path}"
        )

    print(f"Found {len(s25_files)} s25 evaluation files:")
    for f in s25_files:
        print(f"  - {os.path.basename(f)}")

    return s25_files


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


def get_all_calibration_files(participant_folder: str) -> List[str]:
    """Get all calibration files (open_1-4, close_1-4)."""
    calib_files = []
    for set_num in range(1, 5):
        open_file = glob.glob(os.path.join(participant_folder, f"*_open_{set_num}.csv"))
        close_file = glob.glob(os.path.join(participant_folder, f"*_close_{set_num}.csv"))
        if len(open_file) == 1 and len(close_file) == 1:
            calib_files.extend([open_file[0], close_file[0]])
    return calib_files


def train_with_extended_epochs(
    participant: str,
    participant_folder: str,
    variant: str,
    best_config: Dict,
    pretrained_checkpoint: str,
    extended_multiplier: int = 10,
) -> str:
    """
    Train model for extended epochs, saving checkpoint at every epoch.

    Args:
        participant: Participant ID
        participant_folder: Path to participant data
        variant: Fine-tuning variant
        best_config: Best hyperparameters
        pretrained_checkpoint: Path to pretrained checkpoint
        extended_multiplier: Multiply epochs by this factor (default 10)

    Returns:
        Directory containing all epoch checkpoints
    """
    print(f"\n{'='*80}")
    print(f"Extended Training: {participant} - {variant}")
    print(f"{'='*80}\n")

    # Calculate extended epochs
    base_epochs = best_config['epochs']
    extended_epochs = base_epochs * extended_multiplier

    print(f"Base epochs: {base_epochs}")
    print(f"Extended epochs: {extended_epochs}")

    # Get calibration files
    calib_files = get_all_calibration_files(participant_folder)
    print(f"Training on {len(calib_files)} calibration files")

    # Create training folder
    train_dir = os.path.join("temp_convergence_training", f"{participant}_{variant}")
    os.makedirs(train_dir, exist_ok=True)

    # Symlink calibration files
    for calib_file in calib_files:
        link_path = os.path.join(train_dir, os.path.basename(calib_file))
        # Use lexists() to detect broken symlinks (exists() returns False for broken symlinks)
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(calib_file, link_path)

    # Build training command
    exp_name = f"{participant}_{variant}_convergence"

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
        "--exp_name", exp_name,
        "--custom_data_folder", train_dir,
        "--save_every_epoch", "1",  # Per-epoch checkpointing (already default behavior)
    ]

    # Add variant-specific flags
    if variant == 'stroke_only':
        pass  # Train from scratch, no pretrained checkpoint
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
    print("\nStarting extended training...")
    subprocess.run(cmd, check=True)

    # Find checkpoint directory
    # Sort by name descending to get the most recent (timestamps are in YYYY-MM-DD_HH-MM-SS format)
    checkpoint_dir_pattern = f"model_checkpoints/{exp_name}_*"
    checkpoint_dirs = sorted(glob.glob(checkpoint_dir_pattern), reverse=True)
    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"No checkpoint directory found for {exp_name}")

    checkpoint_dir = checkpoint_dirs[0]
    if len(checkpoint_dirs) > 1:
        print(f"Note: Found {len(checkpoint_dirs)} checkpoint directories matching '{exp_name}_*', "
              f"using most recent: {os.path.basename(checkpoint_dir)}")

    # Copy all epoch checkpoints to organized location
    convergence_checkpoint_dir = f"model_checkpoints/convergence/{participant}"
    os.makedirs(convergence_checkpoint_dir, exist_ok=True)

    import shutil
    # Epochs are 0-indexed (0 to extended_epochs-1)
    for epoch in range(extended_epochs):
        src_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        if os.path.exists(src_checkpoint):
            dst_checkpoint = os.path.join(convergence_checkpoint_dir, f"epoch_{epoch}.pth")
            shutil.copy(src_checkpoint, dst_checkpoint)
            print(f"Copied checkpoint for epoch {epoch}")

    print(f"\nCheckpoints saved to: {convergence_checkpoint_dir}")

    return convergence_checkpoint_dir


def evaluate_epoch_checkpoint(
    participant: str,
    participant_folder: str,
    checkpoint_path: str,
    epoch: int,
    s25_files: List[str],
) -> Dict:
    """
    Evaluate a single epoch checkpoint on stroke and healthy data.

    Returns:
        Dict with stroke and healthy metrics
    """
    print(f"\n--- Evaluating Epoch {epoch} ---")

    results = {
        'epoch': epoch,
        'stroke_results': {},
        'healthy_results': {},
    }

    # Evaluate on stroke test sets
    print("Evaluating on stroke test sets...")
    stroke_trans_accs = []
    stroke_raw_accs = []
    stroke_latencies = []

    for condition in TEST_CONDITIONS.keys():
        test_files = get_test_files(participant_folder, condition)
        if len(test_files) == 0:
            continue

        metrics = evaluate_checkpoint_programmatic(
            checkpoint_path=checkpoint_path,
            csv_files=test_files,
            buffer_range=800,
            lookahead=100,
            samples_between_prediction=100,
            allow_relax=1,
            stride=1,
            model_choice="any2any",
            verbose=0,
            compute_latency=True,
        )

        # Convert latency from timesteps to ms (200 Hz = 5ms per timestep)
        latency_ms = float(metrics.get('average_latency', 0)) * 5.0

        results['stroke_results'][condition] = {
            'transition_accuracy': float(metrics['transition_accuracy']),
            'raw_accuracy': float(metrics['raw_accuracy']),
            'avg_detection_latency_ms': latency_ms,
        }

        stroke_trans_accs.append(metrics['transition_accuracy'])
        stroke_raw_accs.append(metrics['raw_accuracy'])
        stroke_latencies.append(latency_ms)

        print(f"  {condition}: Trans={metrics['transition_accuracy']:.4f}, Latency={latency_ms:.1f}ms")

    # Compute average stroke metrics
    results['stroke_avg_transition_acc'] = float(np.mean(stroke_trans_accs)) if stroke_trans_accs else 0.0
    results['stroke_avg_raw_acc'] = float(np.mean(stroke_raw_accs)) if stroke_raw_accs else 0.0
    results['stroke_avg_latency_ms'] = float(np.mean(stroke_latencies)) if stroke_latencies else 0.0

    # Evaluate on healthy s25 data
    print("Evaluating on healthy s25 data...")
    healthy_metrics = evaluate_checkpoint_programmatic(
        checkpoint_path=checkpoint_path,
        csv_files=s25_files,
        buffer_range=800,
        lookahead=100,
        samples_between_prediction=100,
        allow_relax=1,
        stride=1,
        model_choice="any2any",
        verbose=0,
        compute_latency=True,
    )

    # Convert latency from timesteps to ms (200 Hz = 5ms per timestep)
    healthy_latency_ms = float(healthy_metrics.get('average_latency', 0)) * 5.0

    results['healthy_results'] = {
        'transition_accuracy': float(healthy_metrics['transition_accuracy']),
        'raw_accuracy': float(healthy_metrics['raw_accuracy']),
        'avg_detection_latency_ms': healthy_latency_ms,
    }

    print(f"  s25: Trans={healthy_metrics['transition_accuracy']:.4f}, Latency={healthy_latency_ms:.1f}ms")

    print(f"Epoch {epoch} - Stroke Avg: {results['stroke_avg_transition_acc']:.4f} (Latency: {results['stroke_avg_latency_ms']:.1f}ms), Healthy: {results['healthy_results']['transition_accuracy']:.4f}")

    return results


def evaluate_frozen_baseline(
    participant: str,
    participant_folder: str,
    pretrained_checkpoint: str,
    s25_files: List[str],
) -> Dict:
    """Evaluate frozen pretrained model as baseline."""
    print(f"\n{'='*80}")
    print("Evaluating Frozen Pretrained Model (Baseline)")
    print(f"{'='*80}\n")

    return evaluate_epoch_checkpoint(
        participant=participant,
        participant_folder=participant_folder,
        checkpoint_path=pretrained_checkpoint,
        epoch=-1,  # Use -1 to distinguish from trained epoch 0
        s25_files=s25_files,
    )


def run_convergence_study(
    participant: str,
    participant_folder: str,
    variant: str,
    best_config: Dict,
    pretrained_checkpoint: str,
    healthy_s25_path: str = None,
):
    """
    Run complete convergence study for one participant.

    Args:
        participant: Participant ID
        participant_folder: Path to participant data
        variant: Best fine-tuning variant
        best_config: Best hyperparameters
        pretrained_checkpoint: Path to pretrained checkpoint
        healthy_s25_path: Path to healthy s25 data (optional)
    """
    print(f"\n{'='*80}")
    print(f"Convergence Study: {participant} - {variant}")
    print(f"{'='*80}\n")

    # Get healthy s25 files
    s25_files = get_healthy_s25_files(healthy_s25_path)

    # Evaluate frozen baseline
    frozen_baseline_results = evaluate_frozen_baseline(
        participant=participant,
        participant_folder=participant_folder,
        pretrained_checkpoint=pretrained_checkpoint,
        s25_files=s25_files,
    )

    # Save frozen baseline
    frozen_results_dir = f"results/convergence/{participant}/frozen_baseline"
    os.makedirs(frozen_results_dir, exist_ok=True)
    with open(os.path.join(frozen_results_dir, "metrics.json"), 'w') as f:
        json.dump(frozen_baseline_results, f, indent=4)

    # Train with extended epochs
    checkpoint_dir = train_with_extended_epochs(
        participant=participant,
        participant_folder=participant_folder,
        variant=variant,
        best_config=best_config,
        pretrained_checkpoint=pretrained_checkpoint,
        extended_multiplier=10,
    )

    # Evaluate each epoch checkpoint (epochs are 0-indexed)
    extended_epochs = best_config['epochs'] * 10
    all_epoch_results = []

    for epoch in range(extended_epochs):
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for epoch {epoch}")
            continue

        epoch_results = evaluate_epoch_checkpoint(
            participant=participant,
            participant_folder=participant_folder,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            s25_files=s25_files,
        )

        all_epoch_results.append(epoch_results)

        # Save individual epoch results
        epoch_results_dir = f"results/convergence/{participant}/epoch_{epoch}"
        os.makedirs(epoch_results_dir, exist_ok=True)
        with open(os.path.join(epoch_results_dir, "metrics.json"), 'w') as f:
            json.dump(epoch_results, f, indent=4)

    # Save convergence curves data
    curves_file = f"results/convergence/{participant}/convergence_curves.json"
    with open(curves_file, 'w') as f:
        json.dump({
            'participant': participant,
            'variant': variant,
            'base_epochs': best_config['epochs'],
            'extended_epochs': extended_epochs,
            'frozen_baseline': frozen_baseline_results,
            'epoch_results': all_epoch_results,
        }, f, indent=4)

    print(f"\nConvergence curves saved to: {curves_file}")
    print(f"\nConvergence study complete for {participant}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convergence Study")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., p15)")
    parser.add_argument("--variant", required=True, help="Best fine-tuning variant")
    parser.add_argument("--config_file", required=True, help="Path to best config JSON file")
    parser.add_argument("--healthy_s25_path",
                       default="~/Workspace/reactemg/data/ROAM_EMG/s25",
                       help="Path to healthy s25 data for catastrophic forgetting evaluation")

    args = parser.parse_args()

    participant = args.participant
    participant_folder = os.path.expanduser(PARTICIPANTS[participant])

    # Load best config
    with open(args.config_file, 'r') as f:
        config_data = json.load(f)
        best_config = config_data['best_config']

    run_convergence_study(
        participant=participant,
        participant_folder=participant_folder,
        variant=args.variant,
        best_config=best_config,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        healthy_s25_path=args.healthy_s25_path,
    )

    print("\nConvergence study complete!")
