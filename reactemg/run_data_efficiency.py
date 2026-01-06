"""
Data efficiency experiment for ReactEMG stroke.

This script evaluates how model performance changes with different amounts
of calibration data (K paired repetitions where K ∈ {1, 4, 8}).

Uses the best fine-tuning strategy identified from the main experiment.
"""

import os
import sys
import json
import numpy as np
import subprocess
import glob
from typing import Dict, List, Tuple
from dataset_utils import get_paired_repetition_indices, sample_repetitions
from event_classification import evaluate_checkpoint_programmatic


# Configuration
PARTICIPANTS = {
    'p15': '~/Workspace/myhand/src/collected_data/2025_12_04',
    'p20': '~/Workspace/myhand/src/collected_data/2025_12_18',
}

PRETRAINED_CHECKPOINT = "/home/rsw1/Workspace/reactemg/reactemg/model_checkpoints/LOSO_s14_left_2025-11-15_19-01-41_pc1/epoch_4.pth"

TEST_CONDITIONS = {
    'mid_session_baseline': ['open_5.csv', 'close_5.csv'],
    'end_session_baseline': ['open_fatigue.csv', 'close_fatigue.csv'],
    'unseen_posture': ['open_hovering.csv', 'close_hovering.csv'],
    'sensor_shift': ['open_sensor_shift.csv', 'close_sensor_shift.csv'],
    'orthosis_actuated': ['close_from_open.csv'],
}


def get_test_files(participant_folder: str, condition: str) -> List[str]:
    """Get test files for a specific condition."""
    file_patterns = TEST_CONDITIONS[condition]
    test_files = []
    for pattern in file_patterns:
        files = glob.glob(os.path.join(participant_folder, f"*_{pattern}"))
        if len(files) == 1:
            test_files.append(files[0])
    return test_files


def train_with_sampled_data(
    participant: str,
    variant: str,
    best_config: Dict,
    sampled_g_names: List[str],
    paired_reps: Dict,
    budget_k: int,
    trial_idx: int,
    pretrained_checkpoint: str,
) -> str:
    """
    Train model with sampled repetitions.

    Args:
        participant: Participant ID
        variant: Fine-tuning variant
        best_config: Best hyperparameters from main experiment
        sampled_g_names: List of g_i names to use (e.g., ['g_0', 'g_3', 'g_5'])
        paired_reps: Dict from get_paired_repetition_indices()
        budget_k: Number of paired reps (1, 4, or 8)
        trial_idx: Trial index (0-11)
        pretrained_checkpoint: Path to pretrained checkpoint

    Returns:
        Path to saved checkpoint
    """
    # Build sampled_segments dict for dataset
    sampled_segments = {}

    for g_name in sampled_g_names:
        open_file, close_file, open_seg_list, close_seg_list = paired_reps[g_name]

        # Add segments for open file
        if open_file not in sampled_segments:
            sampled_segments[open_file] = []
        sampled_segments[open_file].extend(open_seg_list)

        # Add segments for close file
        if close_file not in sampled_segments:
            sampled_segments[close_file] = []
        sampled_segments[close_file].extend(close_seg_list)

    # Save sampled_segments to a JSON file for the training script to load
    # Create unique temp directory to avoid race conditions
    import tempfile
    import uuid
    temp_trial_id = uuid.uuid4().hex[:8]
    temp_segments_dir = os.path.join(
        tempfile.gettempdir(),
        f"reactemg_segments_{participant}_{variant}_K{budget_k}_trial{trial_idx}_{temp_trial_id}"
    )
    os.makedirs(temp_segments_dir, exist_ok=True)
    segments_file = os.path.join(temp_segments_dir, "sampled_segments.json")
    with open(segments_file, 'w') as f:
        # Convert to serializable format
        segments_serializable = {}
        for file_path, seg_list in sampled_segments.items():
            segments_serializable[file_path] = [(int(s), int(e)) for s, e in seg_list]
        json.dump(segments_serializable, f)

    # Create experiment name
    exp_name = f"{participant}_data_efficiency_{variant}_K{budget_k}_trial{trial_idx}"

    # Build training command
    # Note: We need to modify the training pipeline to accept sampled_segments
    # For now, we'll use a custom approach

    # Create a custom training script call that handles sampled_segments
    # This requires modifying preprocessing_utils.py to accept sampled_segments
    # For this implementation, we'll use a workaround: create temporary CSV files

    # Workaround: Create temporary CSV files with extracted segments
    temp_train_dir = os.path.join("temp_data_efficiency_training", exp_name)
    os.makedirs(temp_train_dir, exist_ok=True)

    # Extract segments and save to temporary CSV files
    import pandas as pd
    file_counter = 0

    for file_path, seg_list in sampled_segments.items():
        df = pd.read_csv(file_path)

        for seg_idx, (start_idx, end_idx) in enumerate(seg_list):
            segment_df = df.iloc[start_idx:end_idx].copy()

            # Save to temporary file
            base_name = os.path.basename(file_path).replace('.csv', '')
            temp_file = os.path.join(temp_train_dir, f"{base_name}_seg{file_counter}.csv")
            segment_df.to_csv(temp_file, index=False)
            file_counter += 1

    # Build training command with temporary files
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
        "--custom_data_folder", temp_train_dir,
    ]

    # Add variant-specific flags
    if variant == 'head_only':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
        cmd.extend(["--freeze_backbone", "1"])
    elif variant == 'lora':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
        cmd.extend(["--use_lora", "1", "--lora_rank", "16", "--lora_alpha", "8", "--lora_dropout_p", "0.05"])
    elif variant == 'full_finetune':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
    # stroke_only would omit checkpoint

    # Run training
    print(f"\nTraining {exp_name}...")
    subprocess.run(cmd, check=True)

    # Find checkpoint
    checkpoint_dir = f"model_checkpoints/{exp_name}_*"
    checkpoint_dirs = glob.glob(checkpoint_dir)
    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"No checkpoint found for {exp_name}")

    checkpoint_dir = checkpoint_dirs[0]
    # Epochs are 0-indexed, so final epoch is epochs-1
    final_epoch = best_config['epochs'] - 1
    epoch_files = glob.glob(os.path.join(checkpoint_dir, f"epoch_{final_epoch}.pth"))
    if len(epoch_files) == 0:
        raise FileNotFoundError(f"No epoch_{final_epoch}.pth found")

    checkpoint_path = epoch_files[0]

    # Copy to organized location
    final_checkpoint_dir = f"model_checkpoints/data_efficiency/{participant}"
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(final_checkpoint_dir, f"{variant}_K{budget_k}_trial{trial_idx}.pth")

    import shutil
    shutil.copy(checkpoint_path, final_checkpoint_path)

    # Clean up temporary files
    shutil.rmtree(temp_segments_dir)
    shutil.rmtree(temp_train_dir)

    return final_checkpoint_path


def evaluate_on_all_conditions(
    participant: str,
    participant_folder: str,
    checkpoint_path: str,
    budget_k: int,
    trial_idx: int,
):
    """Evaluate checkpoint on all test conditions."""

    results = {}

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
        )

        results[condition] = {
            'transition_accuracy': float(metrics['transition_accuracy']),
            'raw_accuracy': float(metrics['raw_accuracy']),
        }

        print(f"  {condition}: Trans={metrics['transition_accuracy']:.4f}, Raw={metrics['raw_accuracy']:.4f}")

    return results


def run_data_efficiency_experiment(
    participant: str,
    participant_folder: str,
    variant: str,
    best_config: Dict,
    pretrained_checkpoint: str,
):
    """
    Run complete data efficiency experiment for one participant.

    Args:
        participant: Participant ID (e.g., 'p15')
        participant_folder: Path to participant data
        variant: Best fine-tuning variant from main experiment
        best_config: Best hyperparameters from main experiment
        pretrained_checkpoint: Path to pretrained checkpoint
    """
    print(f"\n{'='*80}")
    print(f"Data Efficiency Experiment: {participant} - {variant}")
    print(f"{'='*80}\n")

    # Get paired repetition indices
    paired_reps = get_paired_repetition_indices(
        participant_folder=participant_folder,
        num_sets=4,
        reps_per_set=3,
    )

    print(f"Found {len(paired_reps)} paired repetitions (g_0 through g_11)")

    # For each budget K
    for budget_k in [1, 4, 8]:
        print(f"\n{'*'*80}")
        print(f"Budget K = {budget_k}")
        print(f"{'*'*80}\n")

        # Sample repetitions for this budget
        sampled_trials = sample_repetitions(
            paired_reps=paired_reps,
            budget_k=budget_k,
            num_trials=12,
            seed=42,
        )

        print(f"Generated {len(sampled_trials)} trials for K={budget_k}")

        # Store results across trials
        all_trial_results = []

        # For each trial
        for trial_idx, sampled_g_names in enumerate(sampled_trials):
            print(f"\n--- Trial {trial_idx + 1}/12 ---")
            print(f"Using repetitions: {sampled_g_names}")

            # Train model
            checkpoint_path = train_with_sampled_data(
                participant=participant,
                variant=variant,
                best_config=best_config,
                sampled_g_names=sampled_g_names,
                paired_reps=paired_reps,
                budget_k=budget_k,
                trial_idx=trial_idx,
                pretrained_checkpoint=pretrained_checkpoint,
            )

            # Evaluate on all test conditions
            print(f"\nEvaluating trial {trial_idx + 1}:")
            trial_results = evaluate_on_all_conditions(
                participant=participant,
                participant_folder=participant_folder,
                checkpoint_path=checkpoint_path,
                budget_k=budget_k,
                trial_idx=trial_idx,
            )

            # Save individual trial results
            trial_results_dir = f"results/data_efficiency/{participant}/K{budget_k}/trial_{trial_idx}"
            os.makedirs(trial_results_dir, exist_ok=True)

            with open(os.path.join(trial_results_dir, "metrics.json"), 'w') as f:
                json.dump({
                    'participant': participant,
                    'variant': variant,
                    'budget_k': budget_k,
                    'trial_idx': trial_idx,
                    'sampled_repetitions': sampled_g_names,
                    'results': trial_results,
                }, f, indent=4)

            all_trial_results.append(trial_results)

        # Aggregate results across trials
        aggregated_results = {}
        for condition in TEST_CONDITIONS.keys():
            trans_accs = [r[condition]['transition_accuracy'] for r in all_trial_results if condition in r]
            raw_accs = [r[condition]['raw_accuracy'] for r in all_trial_results if condition in r]

            if len(trans_accs) > 0:
                aggregated_results[condition] = {
                    'transition_accuracy_mean': float(np.mean(trans_accs)),
                    'transition_accuracy_std': float(np.std(trans_accs)),
                    'raw_accuracy_mean': float(np.mean(raw_accs)),
                    'raw_accuracy_std': float(np.std(raw_accs)),
                }

        # Save aggregated results
        agg_file = f"results/data_efficiency/{participant}/K{budget_k}/aggregated_metrics.json"
        os.makedirs(os.path.dirname(agg_file), exist_ok=True)

        with open(agg_file, 'w') as f:
            json.dump({
                'participant': participant,
                'variant': variant,
                'budget_k': budget_k,
                'num_trials': len(sampled_trials),
                'aggregated_results': aggregated_results,
            }, f, indent=4)

        print(f"\n{'*'*80}")
        print(f"K={budget_k} Complete - Aggregated Results:")
        for condition, metrics in aggregated_results.items():
            print(f"  {condition}:")
            print(f"    Trans Acc: {metrics['transition_accuracy_mean']:.4f} ± {metrics['transition_accuracy_std']:.4f}")
            print(f"    Raw Acc: {metrics['raw_accuracy_mean']:.4f} ± {metrics['raw_accuracy_std']:.4f}")
        print(f"{'*'*80}\n")

    print(f"\nData efficiency experiment complete for {participant}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Efficiency Experiment")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., p15)")
    parser.add_argument("--variant", required=True, help="Best fine-tuning variant")
    parser.add_argument("--config_file", required=True, help="Path to best config JSON file")

    args = parser.parse_args()

    participant = args.participant
    participant_folder = os.path.expanduser(PARTICIPANTS[participant])

    # Load best config
    with open(args.config_file, 'r') as f:
        config_data = json.load(f)
        best_config = config_data['best_config']

    run_data_efficiency_experiment(
        participant=participant,
        participant_folder=participant_folder,
        variant=args.variant,
        best_config=best_config,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
    )

    print("\nData efficiency experiment complete!")
