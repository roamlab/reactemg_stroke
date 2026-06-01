# ReactEMG-Stroke: Healthy-to-Stroke Transfer Learning for sEMG Intent Detection

This repository extends [ReactEMG](https://github.com/roamlab/reactemg) to study **few-shot adaptation** of healthy-pretrained sEMG models to stroke survivors. It provides a systematic experimental framework for comparing fine-tuning strategies when adapting a model trained on healthy subjects to stroke participants with limited calibration data.

> **Prerequisites**: This repository builds on ReactEMG. For installation instructions, model architecture details, and background on the Any2Any transformer, see the [ReactEMG README](https://github.com/roamlab/reactemg#readme).

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
│  ├── Search: 27 configs (3 LRs × 3 epochs × 3 dropouts)                 │
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

### Prerequisites & setup

Before running anything, make sure the pipeline can find your data and the
healthy-pretrained model — **neither lives in this repo**:

1. **Point the scripts at your paths.** `run_main_experiment.py`,
   `run_data_efficiency.py`, and `run_convergence.py` each define two values
   hard-coded near the top of the file:
   - `PARTICIPANTS` — maps each participant to its data folder (the `open_*/close_*`
     CSVs described above).
   - `PRETRAINED_CHECKPOINT` — the healthy-pretrained Any2Any checkpoint that every
     fine-tuned variant adapts from (produced by base ReactEMG, e.g. a healthy LOSO run).

   Edit these to match your machine.

2. **Disable / configure Weights & Biases.** Every training run calls `wandb.init`.
   To reproduce without a W&B account:
   ```bash
   export WANDB_MODE=disabled
   ```

**Recommended order for a full from-scratch reproduction.** Steps 2–3 reuse the
per-variant CV configs that step 1 writes to `temp_cv_checkpoints/`, so run step 1 first:

| Step | Command | Produces |
|------|---------|----------|
| 1 | `python3 run_main_experiment.py --participant all` | Table 2 results **and** the per-variant CV configs in `temp_cv_checkpoints/` (reused below) |
| 2 | `run_data_efficiency.py` for all variants — see §2 | Data-efficiency results |
| 3 | `run_convergence.py` for all variants — see §3 | Convergence results |
| 4 | analysis scripts — see §4 | The paper's tables and figures from the JSON under `results/` |

### 1. Main Experiment (Full Pipeline)

The main experiment script orchestrates zero-shot evaluation, hyperparameter search, final training, and evaluation for all strategies.

```bash
python3 run_main_experiment.py --participant all
```

This orchestrates:
- Zero-shot evaluation on stroke data
- 4-fold CV hyperparameter search per strategy — 27 configs (LR {5e-5, 1e-4, 5e-4} × epochs {5, 10, 15} × dropout {0, 0.1, 0.2}) × 4 folds = 108 runs
- Final training with best hyperparameters (saved to `temp_cv_checkpoints/{participant}_{variant}_cv_results.json`)
- Evaluation on all 5 test conditions

### 2. Data Efficiency Experiment

Evaluates performance with limited calibration data (K = 1, 4, 8 paired repetitions, 12 trials per K). Each run reuses the CV config the main experiment wrote for that variant. Run all subjects and all variants:

```bash
for v in stroke_only head_only lora full_finetune; do
  python3 run_data_efficiency.py --participant all --variant "$v"
done
```

**Sampling**: K=1 uses one unique repetition per trial (trial *i* uses g_*i*); K>1 samples K of the 12 repetitions without replacement per trial.

### 3. Convergence Study

Trains for a **fixed 100 epochs** (far beyond the CV-selected optimum), evaluating every 5 epochs — 21 checkpoints — on the stroke test sets to track learning dynamics. Run all subjects and all variants:

```bash
for v in stroke_only head_only lora full_finetune; do
  python3 run_convergence.py --participant all --variant "$v"
done
```

### 4. Generating Tables & Figures

With `results/` populated, these scripts produce the paper's tables and figures (subject mapping **p4 = S1, p15 = S2, p20 = S3**):

```bash
python3 extract_results.py                                              # Table 2      -> results/main_experiment/table2.txt
python3 plot_main_results.py                                           # Table 2 bars -> results/main_experiment/table2_bars.png
python3 analyze_data_efficiency.py --compare -o figure_dataeff.png     # data-efficiency figure
python3 analyze_convergence.py --combined -p p15 -o figure_conv_s2.png # convergence figure (per subject; p15 = S2)
```

- The `--compare` and `--combined` figures require the corresponding experiment to have been run for **every overlaid variant** (data efficiency defaults to `head_only lora full_finetune`; convergence needs all four).
- Pass `--variant <v> --participant <p>` to either `analyze_*` script for a single numeric summary, or `-o <path>` to set the output file.

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

Refer to the [ReactEMG README](https://github.com/roamlab/reactemg#readme) for how these parameters shape the online smoothing behavior and the transition-accuracy metric.

## Contact

For questions or support, please email Runsheng at runsheng.w@columbia.edu

## License

This project is released under the MIT License; see the [LICENSE](LICENSE) file for details.
