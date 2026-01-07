"""
Cross-validation hyperparameter search for stroke EMG experiments.

This script performs 4-fold cross-validation over the calibration pool
to select the best hyperparameters for each fine-tuning strategy.
"""

import os
import sys
import glob
import itertools
import subprocess
import json
import numpy as np
from typing import Dict, List, Tuple
from event_classification import evaluate_checkpoint_programmatic


def generate_hyperparameter_configs(variant: str) -> List[Dict]:
    """
    Generate all hyperparameter configurations for a given variant.

    Args:
        variant: One of ['stroke_only', 'head_only', 'lora', 'full_finetune']

    Returns:
        List of hyperparameter dictionaries
    """
    # Common search space
    learning_rates = [5e-5, 1e-4, 5e-4]
    epochs_list = [5, 10, 15]
    dropouts = [0, 0.1, 0.2]

    # Generate all combinations
    configs = []
    for lr, epochs, dropout in itertools.product(learning_rates, epochs_list, dropouts):
        config = {
            'learning_rate': lr,
            'epochs': epochs,
            'dropout': dropout,
            'variant': variant,
        }
        configs.append(config)

    print(f"Generated {len(configs)} configurations for {variant}")
    return configs


def get_fold_files(participant_folder: str, fold_idx: int) -> Tuple[List[str], List[str]]:
    """
    Get training and validation files for a specific CV fold.

    Args:
        participant_folder: Path to participant data folder
        fold_idx: Fold index (0-3)

    Returns:
        (train_files, val_files) where each is a list of file paths
    """
    # All baseline files
    all_sets = []
    for set_num in range(1, 5):
        open_file = glob.glob(os.path.join(participant_folder, f"*_open_{set_num}.csv"))
        close_file = glob.glob(os.path.join(participant_folder, f"*_close_{set_num}.csv"))
        if len(open_file) == 1 and len(close_file) == 1:
            all_sets.append((set_num, open_file[0], close_file[0]))

    if len(all_sets) != 4:
        raise ValueError(f"Expected 4 baseline sets, found {len(all_sets)}")

    # Validation set is fold_idx
    val_set = all_sets[fold_idx]
    val_files = [val_set[1], val_set[2]]  # open and close files

    # Training sets are all others
    train_files = []
    for i, (set_num, open_f, close_f) in enumerate(all_sets):
        if i != fold_idx:
            train_files.extend([open_f, close_f])

    return train_files, val_files


def run_training(
    config: Dict,
    train_files: List[str],
    participant: str,
    fold_idx: int,
    pretrained_checkpoint: str,
    temp_dir: str,
) -> str:
    """
    Train a model with specified hyperparameters.

    Args:
        config: Hyperparameter configuration
        train_files: List of training file paths
        participant: Participant ID (e.g., 'p15')
        fold_idx: Fold index
        pretrained_checkpoint: Path to pretrained checkpoint
        temp_dir: Directory for temporary checkpoints

    Returns:
        Path to saved checkpoint
    """
    variant = config['variant']
    lr = config['learning_rate']
    epochs = config['epochs']
    dropout = config['dropout']

    # Create temporary folder for this fold's training data
    fold_train_dir = os.path.join(temp_dir, f"{participant}_{variant}_fold{fold_idx}_train")
    os.makedirs(fold_train_dir, exist_ok=True)

    # Create symlinks to training files in temp folder
    for train_file in train_files:
        link_path = os.path.join(fold_train_dir, os.path.basename(train_file))
        # Use lexists() to detect broken symlinks (exists() returns False for broken symlinks)
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(train_file, link_path)

    # Construct experiment name
    exp_name = f"{participant}_{variant}_fold{fold_idx}_lr{lr}_ep{epochs}_do{dropout}"

    # Build training command
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
        "--learning_rate", str(lr),
        "--epochs", str(epochs),
        "--dropout", str(dropout),
        "--exp_name", exp_name,
        "--custom_data_folder", fold_train_dir,
    ]

    # Add variant-specific flags
    if variant == 'stroke_only':
        # Train from scratch (no pretrained checkpoint)
        pass
    elif variant == 'head_only':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
        cmd.extend(["--freeze_backbone", "1"])
    elif variant == 'lora':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
        cmd.extend(["--use_lora", "1"])
        cmd.extend(["--lora_rank", "16"])
        cmd.extend(["--lora_alpha", "8"])
        cmd.extend(["--lora_dropout_p", "0.05"])
    elif variant == 'full_finetune':
        cmd.extend(["--saved_checkpoint_pth", pretrained_checkpoint])
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Run training
    print(f"\n{'='*80}")
    print(f"Training: {exp_name}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, check=True)

    # Find the saved checkpoint
    # Sort by name descending to get the most recent (timestamps are in YYYY-MM-DD_HH-MM-SS format)
    checkpoint_dir_pattern = f"model_checkpoints/{exp_name}_*"
    checkpoint_dirs = sorted(glob.glob(checkpoint_dir_pattern), reverse=True)
    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"No checkpoint found matching pattern: {checkpoint_dir_pattern}")

    checkpoint_dir = checkpoint_dirs[0]
    if len(checkpoint_dirs) > 1:
        print(f"Note: Found {len(checkpoint_dirs)} checkpoint directories matching '{exp_name}_*', "
              f"using most recent: {os.path.basename(checkpoint_dir)}")
    # Find epoch checkpoint (use final epoch - epochs are 0-indexed, so final is epochs-1)
    epoch_files = glob.glob(os.path.join(checkpoint_dir, f"epoch_{epochs-1}.pth"))
    if len(epoch_files) == 0:
        raise FileNotFoundError(f"No epoch_{epochs-1}.pth found in {checkpoint_dir}")

    checkpoint_path = epoch_files[0]
    return checkpoint_path


def evaluate_on_validation(checkpoint_path: str, val_files: List[str]) -> float:
    """
    Evaluate a checkpoint on validation files and return transition accuracy.

    Args:
        checkpoint_path: Path to model checkpoint
        val_files: List of validation file paths

    Returns:
        Transition accuracy (float)
    """
    print(f"\nEvaluating checkpoint on {len(val_files)} validation files...")

    metrics = evaluate_checkpoint_programmatic(
        checkpoint_path=checkpoint_path,
        csv_files=val_files,
        buffer_range=800,
        lookahead=100,
        samples_between_prediction=100,
        allow_relax=1,
        stride=1,
        model_choice="any2any",
        verbose=0,
    )

    transition_acc = metrics['transition_accuracy']
    print(f"Validation transition accuracy: {transition_acc:.4f}")

    return transition_acc


def hyperparameter_search(
    participant: str,
    participant_folder: str,
    variant: str,
    pretrained_checkpoint: str,
    temp_dir: str = "temp_cv_checkpoints",
) -> Dict:
    """
    Perform 4-fold CV hyperparameter search for a specific variant.

    Args:
        participant: Participant ID (e.g., 'p15')
        participant_folder: Path to participant data folder
        variant: One of ['stroke_only', 'head_only', 'lora', 'full_finetune']
        pretrained_checkpoint: Path to pretrained checkpoint
        temp_dir: Directory for temporary checkpoints

    Returns:
        Best hyperparameter configuration with validation metrics
    """
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search: {participant} - {variant}")
    print(f"{'='*80}\n")

    os.makedirs(temp_dir, exist_ok=True)

    # Generate all configs
    configs = generate_hyperparameter_configs(variant)

    # Store results
    results = []

    # For each config, run 4-fold CV
    for config_idx, config in enumerate(configs):
        print(f"\nConfig {config_idx + 1}/{len(configs)}: {config}")

        fold_accs = []

        for fold_idx in range(4):
            print(f"\n--- Fold {fold_idx + 1}/4 ---")

            # Get train/val files for this fold
            train_files, val_files = get_fold_files(participant_folder, fold_idx)
            print(f"Training files: {len(train_files)}")
            print(f"Validation files: {len(val_files)}")

            # Train model
            checkpoint_path = run_training(
                config=config,
                train_files=train_files,
                participant=participant,
                fold_idx=fold_idx,
                pretrained_checkpoint=pretrained_checkpoint,
                temp_dir=temp_dir,
            )

            # Evaluate on validation set
            val_acc = evaluate_on_validation(checkpoint_path, val_files)
            fold_accs.append(val_acc)

        # Compute average across folds
        avg_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)

        result = {
            'config': config,
            'fold_accs': fold_accs,
            'avg_acc': avg_acc,
            'std_acc': std_acc,
        }
        results.append(result)

        print(f"\nConfig {config_idx + 1} - Avg Acc: {avg_acc:.4f} ± {std_acc:.4f}")

    # Select best config
    # Primary: highest avg_acc
    # Tiebreaker: fewest epochs
    results_sorted = sorted(
        results,
        key=lambda x: (-x['avg_acc'], x['config']['epochs'])
    )

    best_result = results_sorted[0]
    best_config = best_result['config']

    print(f"\n{'='*80}")
    print(f"Best Configuration for {participant} - {variant}:")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Epochs: {best_config['epochs']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Avg Transition Accuracy: {best_result['avg_acc']:.4f} ± {best_result['std_acc']:.4f}")
    print(f"{'='*80}\n")

    # Save results to JSON
    results_file = os.path.join(temp_dir, f"{participant}_{variant}_cv_results.json")
    with open(results_file, 'w') as f:
        # Convert to serializable format
        results_serializable = []
        for r in results:
            results_serializable.append({
                'config': r['config'],
                'fold_accs': [float(x) for x in r['fold_accs']],
                'avg_acc': float(r['avg_acc']),
                'std_acc': float(r['std_acc']),
            })
        json.dump({
            'participant': participant,
            'variant': variant,
            'all_results': results_serializable,
            'best_config': best_config,
            'best_avg_acc': float(best_result['avg_acc']),
            'best_std_acc': float(best_result['std_acc']),
        }, f, indent=4)

    print(f"CV results saved to: {results_file}")

    return best_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CV Hyperparameter Search")
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., p15)")
    parser.add_argument("--participant_folder", required=True, help="Path to participant data")
    parser.add_argument("--variant", required=True,
                        choices=['stroke_only', 'head_only', 'lora', 'full_finetune'],
                        help="Fine-tuning variant")
    parser.add_argument("--pretrained_checkpoint", required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--temp_dir", default="temp_cv_checkpoints",
                        help="Directory for temporary checkpoints")

    args = parser.parse_args()

    best_config = hyperparameter_search(
        participant=args.participant,
        participant_folder=args.participant_folder,
        variant=args.variant,
        pretrained_checkpoint=args.pretrained_checkpoint,
        temp_dir=args.temp_dir,
    )

    print("\nSearch complete!")
