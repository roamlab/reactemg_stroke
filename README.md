# ReactEMG-Stroke: Healthy-to-Stroke Transfer Learning for sEMG Intent Detection

This repository extends [ReactEMG](https://github.com/roamlab/reactemg) to study **few-shot adaptation** of healthy-pretrained sEMG models to stroke survivors. It provides a systematic experimental framework for comparing fine-tuning strategies when adapting a model trained on healthy subjects to stroke participants with limited calibration data.

> **Prerequisites**: This repository builds on ReactEMG. For installation instructions, model architecture details, and background on the Any2Any transformer, see the [ReactEMG README](https://github.com/roamlab/reactemg#readme).

## Participants & Data Structure

### Stroke Participants
- **s1** (data folder: `data/s1`)
- **s2** (data folder: `data/s2`)
- **s3** (data folder: `data/s3`)

### Data Organization

Each participant's data is organized into calibration and test sets:

```
data/s1/                                                 # s2/ and s3/ share this layout
├── s1_open_1.csv, s1_close_1.csv     ┐
├── s1_open_2.csv, s1_close_2.csv     │  Calibration pool
├── s1_open_3.csv, s1_close_3.csv     │  (4 baseline sets × 3 reps each = 12 paired reps)
├── s1_open_4.csv, s1_close_4.csv     ┘
│
├── s1_open_5.csv, s1_close_5.csv                         # mid_session_baseline
├── s1_open_fatigue.csv, s1_close_fatigue.csv             # end_session_baseline
├── s1_open_hovering.csv, s1_close_hovering.csv           # unseen_posture
├── s1_open_sensor_shift.csv, s1_close_sensor_shift.csv   # sensor_shift
└── s1_close_from_open.csv                                # orthosis_actuated
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

The stroke dataset and the healthy-pretrained checkpoint do **not** live in this repo;
set both up before running:

1. **Download the stroke dataset.** It is **not** bundled with this repository. Unpack it
   at the **repository root** so the three subject folders land at `data/s1`, `data/s2`,
   and `data/s3` (the layout shown above). The run scripts resolve `PARTICIPANTS` from this
   location automatically, so there are no data paths to edit in the code. For example,
   from a Dropbox share link:
   ```bash
   # Run from the repository root. Replace the URL with your own Dropbox share link and
   # make sure it ends in `dl=1` (Dropbox serves the raw file with dl=1, a preview with dl=0).
   curl -L "https://www.dropbox.com/s/TODO/stroke_dataset.zip?dl=1" -o stroke_dataset.zip  # TODO: your link
   mkdir -p data && unzip stroke_dataset.zip -d data/   # -> data/s1, data/s2, data/s3
   ```

2. **Set the healthy-pretrained checkpoint.** The healthy-pretrained Any2Any model that
   every fine-tuned variant adapts from (produced by base ReactEMG, e.g. a healthy LOSO
   run) is not in this repo either. In `run_main_experiment.py`, `run_data_efficiency.py`,
   and `run_convergence.py`, replace the `PRETRAINED_CHECKPOINT` placeholder (marked `TODO`
   near the top of each file) with the path to your `.pth` checkpoint.

3. **Disable / configure Weights & Biases.** Every training run calls `wandb.init`.
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

With `results/` populated, these scripts produce the paper's tables and figures (subject mapping **s1 = S1, s2 = S2, s3 = S3**):

```bash
python3 extract_results.py                                              # Table 2      -> results/main_experiment/table2.txt
python3 plot_main_results.py                                           # Table 2 bars -> results/main_experiment/table2_bars.png
python3 analyze_data_efficiency.py --compare -o figure_dataeff.png     # data-efficiency figure
python3 analyze_convergence.py --combined -p s2 -o figure_conv_s2.png # convergence figure (per subject; s2 = S2)
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
