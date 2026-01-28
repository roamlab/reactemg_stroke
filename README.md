# ReactEMG-Stroke: Healthy-to-Stroke Transfer Learning for sEMG Intent Detection

This repository extends [ReactEMG](https://github.com/your-org/reactemg) to study **few-shot adaptation** of healthy-pretrained sEMG models to stroke survivors. It provides a systematic experimental framework for comparing fine-tuning strategies when adapting a model trained on healthy subjects to stroke participants with limited calibration data.

> **Prerequisites**: This repository builds on ReactEMG. For installation instructions, model architecture details, and background on the Any2Any transformer, see the [ReactEMG README](https://github.com/your-org/reactemg#readme).

## Participants & Data Structure

### Stroke Participants
- **p4**: Left-hand stroke survivor (data folder: `2026_01_06`)
- **p15**: Left-hand stroke survivor (data folder: `2025_12_04`)
- **p20**: Left-hand stroke survivor (data folder: `2025_12_18`)

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

### 1. Main Experiment (Full Pipeline)

The main experiment script orchestrates zero-shot evaluation, hyperparameter search, final training, and evaluation for all strategies.

**Run all participants:**
```bash
python3 run_main_experiment.py --participant all
```

**Run a single participant:**
```bash
python3 run_main_experiment.py --participant p4
python3 run_main_experiment.py --participant p15
python3 run_main_experiment.py --participant p20
```

This orchestrates:
- Zero-shot evaluation on stroke data
- 4-fold CV hyperparameter search for each strategy (36 configs × 4 folds = 144 runs per strategy)
- Final training with best hyperparameters
- Evaluation on all 5 test conditions

### 2. Hyperparameter Search (Single Strategy)

Run CV search for a specific participant and strategy. This is useful for running individual strategies in parallel or re-running a specific search.

```bash
python3 cv_hyperparameter_search.py \
  --participant p15 \
  --participant_folder ~/Workspace/myhand/src/collected_data/2025_12_04 \
  --variant lora \
  --pretrained_checkpoint /path/to/healthy_pretrained.pth
```

**Available variants**: `stroke_only`, `head_only`, `lora`, `full_finetune`

**Search Space**:
- Learning rate: [5e-5, 1e-4, 5e-4]
- Epochs: [5, 10, 15]
- Dropout: [0, 0.1, 0.2]

Results saved to: `temp_cv_checkpoints/{participant}_{variant}_cv_results.json`

### 3. Data Efficiency Experiment

Evaluates performance with limited calibration data (K=1, 4, 8 paired repetitions) using 12 independent trials per K.

**Run all participants:**
```bash
python3 run_data_efficiency.py --participant all --variant lora
```

**Run a single participant:**
```bash
python3 run_data_efficiency.py --participant p15 --variant lora
```

**With explicit config file:**
```bash
python3 run_data_efficiency.py \
  --participant p15 \
  --variant lora \
  --config_file temp_cv_checkpoints/p15_lora_cv_results.json
```

**Sampling Strategy**:
- K=1: Each trial uses exactly one unique repetition (trial i uses g_i)
- K>1: Random sampling without replacement across all 12 repetitions

### 4. Convergence Study

Trains for 100 epochs (10× optimal) and evaluates every 5 epochs on both stroke test sets and healthy s25 data to track learning dynamics and catastrophic forgetting.

**Run all participants:**
```bash
python3 run_convergence.py --participant all --variant lora
```

**Run a single participant:**
```bash
python3 run_convergence.py --participant p15 --variant lora
```

**With explicit config file and healthy data path:**
```bash
python3 run_convergence.py \
  --participant p15 \
  --variant lora \
  --config_file temp_cv_checkpoints/p15_lora_cv_results.json \
  --healthy_s25_path ~/Workspace/reactemg/data/ROAM_EMG/s25
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
│   └── {variant}/{participant}/
│       ├── K1/  (12 trials + aggregated_metrics.json)
│       ├── K4/
│       └── K8/
│
└── convergence/
    └── {variant}/{participant}/
        ├── frozen_baseline/
        ├── epoch_*/
        └── convergence_curves.json

model_checkpoints/
├── main_experiment/
│   └── {participant}_{variant}_final.pth
├── data_efficiency/
│   └── {variant}/{participant}/K{budget}_trial{n}.pth
└── convergence/
    └── {variant}/{participant}/epoch_{n}.pth
```

## Key Extensions from Base ReactEMG

| Feature | Description |
|---------|-------------|
| `--freeze_backbone` | Freezes all parameters except action classification head |
| `--custom_data_folder` | Load stroke data from arbitrary directory |
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
