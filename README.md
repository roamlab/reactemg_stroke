# ReactEMG-Stroke: Healthy-to-Stroke Transfer Learning for sEMG Intent Detection

This repository extends [ReactEMG](https://github.com/your-org/reactemg) to study **few-shot adaptation** of healthy-pretrained sEMG models to stroke survivors. It provides a systematic experimental framework for comparing fine-tuning strategies when adapting a model trained on healthy subjects to stroke participants with limited calibration data.

> **Prerequisites**: This repository builds on ReactEMG. For installation instructions, model architecture details, and background on the Any2Any transformer, see the [ReactEMG README](https://github.com/your-org/reactemg#readme).

## Overview

Stroke survivors often exhibit different EMG signal patterns due to motor impairments, making direct application of healthy-trained models suboptimal. This project investigates:

1. **Transfer Learning Efficiency**: How well healthy-pretrained models adapt to stroke data
2. **Fine-Tuning Strategy Comparison**: Zero-shot vs. stroke-only vs. head-only vs. LoRA vs. full fine-tuning
3. **Few-Shot Adaptation**: Performance with minimal calibration data (K=1, 4, 8 repetitions)
4. **Robustness to Domain Shift**: Evaluation across 5 different test conditions
5. **Convergence Dynamics**: Learning curves and catastrophic forgetting analysis

## Participants & Data Structure

### Stroke Participants
- **p15**, **p20**: Left-hand stroke survivors (primary participants)
- **p4**: Additional participant (from 2026_01_06 dataset)

### Data Organization

Each participant's data is organized into calibration and test sets:

```
participant_folder/
├── open_1.csv, close_1.csv     ┐
├── open_2.csv, close_2.csv     │  Calibration pool
├── open_3.csv, close_3.csv     │  (4 baseline sets × 3 reps each = 12 paired reps)
├── open_4.csv, close_4.csv     ┘
│
├── open_5.csv, close_5.csv                    # mid_session_baseline
├── open_fatigue.csv, close_fatigue.csv        # end_session_baseline
├── open_hovering.csv, close_hovering.csv      # unseen_posture
├── open_sensor_shift.csv, close_sensor_shift.csv  # sensor_shift
└── close_from_open.csv                        # orthosis_actuated
```

**Calibration Pool**: 12 paired repetitions (g_0 through g_11) extracted from 4 baseline sets, used for training/validation splits.

**Test Conditions** (5 types):
| Condition | Description |
|-----------|-------------|
| `mid_session_baseline` | Mid-session recordings (open_5, close_5) |
| `end_session_baseline` | Post-fatigue recordings (open_fatigue, close_fatigue) |
| `unseen_posture` | Arm hovering posture (open_hovering, close_hovering) |
| `sensor_shift` | After sensor repositioning (open_sensor_shift, close_sensor_shift) |
| `orthosis_actuated` | Orthosis-driven close motion (close_from_open) |

## Fine-Tuning Strategies

This repository compares 5 adaptation strategies:

| Strategy | Description | Command Flags |
|----------|-------------|---------------|
| **Zero-shot** | Frozen pretrained model (baseline) | No training |
| **Stroke-only** | Train from scratch on stroke data | No `--saved_checkpoint_pth` |
| **Head-only** | Freeze backbone, train classification head | `--freeze_backbone 1` |
| **LoRA** | Low-rank adaptation of linear layers | `--use_lora 1` |
| **Full fine-tune** | Update all parameters | Default behavior |

## Experimental Workflow

The experiments follow a three-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENTAL PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: ZERO-SHOT BASELINE                                            │
│  └── Evaluate pretrained healthy model directly on stroke test sets     │
│                                                                         │
│  Stage 2: HYPERPARAMETER SEARCH + TRAINING                              │
│  ├── 4-fold CV across calibration pool                                  │
│  ├── Search: 36 configs (4 LRs × 3 epochs × 3 dropouts)                 │
│  ├── Select best config per variant (primary: transition accuracy)      │
│  └── Train final model on full calibration pool                         │
│                                                                         │
│  Stage 3: EVALUATION                                                    │
│  └── Test all models on 5 test conditions with latency metrics          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Running the Experiments

All commands assume you're in the `reactemg/` directory.

### 1. Validation Tests

Verify implementations before running experiments:

```bash
python3 test_implementations.py
```

### 2. Main Experiment (Full Pipeline)

Run the complete experiment for all participants and strategies:

```bash
python3 run_main_experiment.py --participant all
```

Or for a specific participant:

```bash
python3 run_main_experiment.py --participant p15
```

This orchestrates:
- Zero-shot evaluation on stroke data
- 4-fold CV hyperparameter search for each strategy (36 configs × 4 folds = 144 runs per strategy)
- Final training with best hyperparameters
- Evaluation on all 5 test conditions

### 3. Hyperparameter Search (Single Strategy)

Run CV search for a specific strategy:

```bash
python3 cv_hyperparameter_search.py \
  --participant p15 \
  --participant_folder ~/Workspace/myhand/src/collected_data/2025_12_04 \
  --variant lora \
  --pretrained_checkpoint /path/to/healthy_pretrained.pth
```

**Available variants**: `stroke_only`, `head_only`, `lora`, `full_finetune`

**Search Space**:
- Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
- Epochs: [5, 10, 15]
- Dropout: [0.1, 0.3, 0.5]

Results saved to: `temp_cv_checkpoints/{participant}_{variant}_cv_results.json`

### 4. Data Efficiency Experiment

Evaluate performance with limited calibration data:

```bash
python3 run_data_efficiency.py \
  --participant p15 \
  --variant lora \
  --config_file temp_cv_checkpoints/p15_lora_cv_results.json
```

Tests K=1, 4, 8 paired repetitions with 12 independent trials per K.

**Sampling Strategy**:
- K=1: Each trial uses exactly one unique repetition (trial i uses g_i)
- K>1: Random sampling without replacement across all 12 repetitions

### 5. Convergence Study

Track learning dynamics and catastrophic forgetting:

```bash
python3 run_convergence.py \
  --participant p15 \
  --variant lora \
  --config_file temp_cv_checkpoints/p15_lora_cv_results.json
```

Trains for 100 epochs (10× optimal), saving checkpoints every epoch. Evaluates every 5 epochs on:
- Stroke test sets (5 conditions)
- Healthy baseline (ROAM-EMG s25 data) to detect catastrophic forgetting

## Training Commands

### Head-Only Fine-Tuning

```bash
python3 main.py \
  --freeze_backbone 1 \
  --saved_checkpoint_pth /path/to/healthy_pretrained.pth \
  --custom_data_folder /path/to/stroke_data \
  --dataset_selection custom_folder \
  --num_classes 3 \
  --num_epochs 10 \
  --learning_rate 1e-4 \
  --exp_name p15_head_only
```

### LoRA Fine-Tuning

```bash
python3 main.py \
  --use_lora 1 \
  --lora_rank 16 \
  --lora_alpha 8 \
  --lora_dropout_p 0.05 \
  --saved_checkpoint_pth /path/to/healthy_pretrained.pth \
  --custom_data_folder /path/to/stroke_data \
  --dataset_selection custom_folder \
  --num_classes 3 \
  --num_epochs 10 \
  --learning_rate 5e-5 \
  --exp_name p15_lora
```

### Full Fine-Tuning

```bash
python3 main.py \
  --saved_checkpoint_pth /path/to/healthy_pretrained.pth \
  --custom_data_folder /path/to/stroke_data \
  --dataset_selection custom_folder \
  --num_classes 3 \
  --num_epochs 10 \
  --learning_rate 5e-5 \
  --exp_name p15_full_finetune
```

### Training from Scratch (Stroke-Only)

```bash
python3 main.py \
  --custom_data_folder /path/to/stroke_data \
  --dataset_selection custom_folder \
  --num_classes 3 \
  --num_epochs 15 \
  --learning_rate 1e-4 \
  --exp_name p15_stroke_only
```

## Evaluation

### Command-Line Evaluation

```bash
python3 event_classification.py \
  --eval_task predict_action \
  --files_or_dirs /path/to/test_csvs \
  --buffer_range 800 \
  --lookahead 100 \
  --samples_between_prediction 100 \
  --allow_relax 1 \
  --stride 1 \
  --likelihood_format logits \
  --maj_vote_range future \
  --saved_checkpoint_pth /path/to/checkpoint.pth \
  --model_choice any2any \
  --compute_latency 1
```

### Programmatic API

```python
from event_classification import evaluate_checkpoint_programmatic

metrics = evaluate_checkpoint_programmatic(
    checkpoint_path='model.pth',
    csv_files=['p15_open_5.csv', 'p15_close_5.csv'],
    buffer_range=800,
    lookahead=100,
    samples_between_prediction=100,
    allow_relax=1,
    stride=1,
    model_choice="any2any",
    compute_latency=True,
)

print(f"Transition Accuracy: {metrics['transition_accuracy']:.4f}")
print(f"Raw Accuracy: {metrics['raw_accuracy']:.4f}")
print(f"Average Latency: {metrics['average_latency']:.1f} samples")
```

## Output Structure

```
results/
├── main_experiment/
│   └── {participant}/
│       ├── zero_shot/
│       ├── stroke_only/
│       ├── head_only/
│       ├── lora/
│       └── full_finetune/
│           └── {test_condition}/
│               └── metrics_summary.json
│
├── data_efficiency/
│   └── {participant}/
│       ├── K1/  (12 trials + aggregated_metrics.json)
│       ├── K4/
│       └── K8/
│
└── convergence/
    └── {participant}/
        ├── epoch_*/
        └── convergence_curves.json

model_checkpoints/
├── main_experiment/
│   └── {participant}_{variant}_final.pth
├── data_efficiency/
│   └── {participant}/K{budget}_trial{n}.pth
└── convergence/
    └── {participant}/epoch_{n}.pth
```

## Key Extensions from Base ReactEMG

| Feature | Description |
|---------|-------------|
| `--freeze_backbone` | Freezes all parameters except action classification head |
| `--custom_data_folder` | Load stroke data from arbitrary directory |
| `evaluate_checkpoint_programmatic()` | Python API for model evaluation with latency metrics |
| `dataset_utils.py` | Repetition extraction and K-budget sampling |
| `cv_hyperparameter_search.py` | 4-fold CV hyperparameter selection |
| `run_main_experiment.py` | Complete experimental pipeline orchestration |
| `run_data_efficiency.py` | Few-shot adaptation experiments |
| `run_convergence.py` | Convergence and forgetting analysis |

## Fixed Evaluation Parameters

All stroke experiments use these evaluation settings for consistency:

| Parameter | Value |
|-----------|-------|
| `buffer_range` | 800 |
| `lookahead` | 100 |
| `samples_between_prediction` | 100 |
| `allow_relax` | 1 |
| `stride` | 1 |
| `likelihood_format` | logits |
| `maj_vote_range` | future |

## Troubleshooting

**"No checkpoint found"**
- Verify training completed successfully
- Check `model_checkpoints/` directory
- Ensure epoch number matches configuration

**"Test data not found"**
- Verify participant data paths in script configuration
- Check file naming patterns (e.g., `p15_open_1.csv`)
- Ensure all baseline and test files exist

**CUDA out of memory**
- Reduce batch size (default: 128)
- Run experiments sequentially
- Consider using LoRA instead of full fine-tuning

## Citation

If you use this codebase, please cite the original ReactEMG paper:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2506.19815,
  doi = {10.48550/ARXIV.2506.19815},
  url = {https://arxiv.org/abs/2506.19815},
  author = {Wang, Runsheng and Zhu, Xinyue and Chen, Ava and Xu, Jingxi and Winterbottom, Lauren and Nilsen, Dawn M. and Stein, Joel and Ciocarlie, Matei},
  title = {ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG},
  publisher = {arXiv},
  year = {2025},
}
```

## Contact

For questions about the stroke adaptation experiments, please email Runsheng at runsheng.w@columbia.edu

## License

This project is released under the MIT License; see the [LICENSE](LICENSE) file for details.
