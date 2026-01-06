# ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG

[Project Page](https://reactemg.github.io/) | [arXiv](https://arxiv.org/abs/2506.19815) | [Video](https://youtu.be/AKT8hMvVCGY)

[Runsheng Wang](http://runshengwang.com/)<sup>1</sup>,
[Xinyue Zhu](https://xinyuezhu.com/)<sup>1</sup>,
[Ava Chen](https://avachen.net/),
[Jingxi Xu](https://jxu.ai/),
[Lauren Winterbottom](https://reactemg.github.io/),
[Dawn M. Nilsen](https://www.vagelos.columbia.edu/profile/dawn-nilsen-edd)<sup>2</sup>,
[Joel Stein](https://www.neurology.columbia.edu/profile/joel-stein-m-d)<sup>2</sup>,
[Matei Ciocarlie](https://www.me.columbia.edu/faculty/matei-ciocarlie)<sup>2</sup>

<sup>1</sup>Equal contribution,
<sup>2</sup>Co-Principal Investigators

Columbia University

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="assets/demo.gif">

ReactEMG is a zero-shot, low-latency EMG framework that segments forearm signals in real time to predict hand gestures at every timestep, delivering calibration-free, high-accuracy intent detection ideal for controlling prosthetic and robotic devices.

## :package: Installation
Clone the repo with `--recurse-submodules` and install our conda (mamba) environment on an Ubuntu machine with a NVIDIA GPU. We use Ubuntu 24.04 LTS and Python 3.11. 

```bash
mamba env create -f environment.yml
```

Install [PyTorch](<https://pytorch.org/get-started/locally/>) in the conda environment, then install wandb via pip:

```bash
pip install wandb
```

Lastly, install minLoRA via:

```bash
cd minLoRA && pip install -e .
```

minLorRA was built for editable install with `setup.py develop`, which is deprecated. Consider enabling `--use-pep517` and use `setuptools ≥ 64` when working with `pip ≥ 25.3`.

## :floppy_disk: Datasets

### 1. ROAM-EMG 
We are open-sourcing our sEMG dataset, **ROAM-EMG**.  
- **Scope:** Using the Thalmic Myo armband, we recorded eight-channel sEMG signals from 28 participants as they performed hand gestures in four arm postures, followed by two grasping tasks and three types of arm movement. Full details of the dataset are provided in our paper and its supplementary materials.
- **Download:** [Dropbox Link](<https://www.dropbox.com/scl/fi/19zvl12vn27wsnzsmw0vx/ROAM_EMG.zip?rlkey=x6gtygdfz24i8efdswr1exii7&st=ljzuoire&dl=0>)  

### 2. Pre-processed public datasets  
For full reproducibility, we also provide pre-processed versions of all public EMG dataset used in the paper. The file structures and data formats have been aligned with ROAM-EMG. We recommend organizing all datasets under the `data/` folder (automatically created with the command below) in the root directory of the repo. To download all datasets (including ROAM-EMG): 

```bash
curl -L -o data.zip "https://www.dropbox.com/scl/fi/isj4450alriqjfstkna2s/data.zip?rlkey=n5sf910lopskewzyae0vgn6j7&st=vt89hfpj&dl=1" && unzip data.zip && rm data.zip
```

## :hammer_and_wrench: Training

### Logging

We use W&B to track experiments. Decide whether you want metrics online or offline:

```bash
# online (default) – set once in your shell
export WANDB_PROJECT=my-emg-project
export WANDB_ENTITY=<your-wandb-username>

# or completely disable
export WANDB_MODE=disabled
```

### Pre-training with public datasets

Use the following command to pre-train our model on EMG-EPN-612 and other public datasets:

```bash
python3 main.py \
  --embedding_method linear_projection \
  --use_input_layernorm \
  --task_selection 0 1 2 \
  --offset 30 \
  --share_pe \
  --num_classes 3 \
  --use_warmup_and_decay \
  --dataset_selection pub_with_epn \
  --window_size 600 \
  --val_patient_ids s1 \
  --epn_subset_percentage 1.0 \
  --model_choice any2any \
  --inner_window_size 600 \
  --exp_name <RUN_ID>
```

Replace <RUN_ID> with your desired name, and the script will save checkpoints to `model_checkpoints/<RUN_ID>_<timestamp>_<machine_name>/epoch_<N>.pth`, where `<timestamp>` records the run’s start time and `<machine_name>` identifies the host. Ensure you have write permission where you launch the job.

You may also initialize weights from a saved checkpoint by adding `--saved_checkpoint_pth path/to/epoch_X.pth` to the training command. If you wish to fine-tune a model via LoRA, provide the flag `--use_lora 1`, in addition to the locally saved checkpoint path.

To train EPN-only models for evaluation purposes, set `--dataset_selection epn_only`

If this is your first time using W&B on your machine, you will be prompted to provide credentials:

```text
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

Enter `2` to use your W&B account, and follow the prompts to provide your API key.

### Fine-tuning on ROAM-EMG

Fine-tuning follows a leave-one-subject-out (LOSO) protocol. The helper script `finetune_runner.sh` trains a separate model for every subject in the ROAM-EMG dataset. Open `finetune_runner.sh` and set `saved_checkpoint_pth` to be your pre-trained checkpoint path, and start LOSO fine-tuning via:

```bash
source finetune_runner.sh
```

## :bar_chart: Evaluation

We evaluate model performance on two metrics:
- **Raw accuracy**: Per-timestep correctness across the entire EMG recording
- **Transition accuracy**: Event-level score that captures accuracy and stability

During evaluation, we run the model exactly as how it would run online: windows slide forward in real time and predictions are aggregated live. This gives a realistic view of online performance instead of an offline, hindsight-only score.

### Run the evaluation
```bash
python3 event_classification.py \
  --eval_task predict_action \
  --files_or_dirs ../data/ROAM_EMG \
  --allow_relax 0 \
  --buffer_range 200 \
  --stride 1 \
  --lookahead 50 \
  --weight_max_factor 1.0 \
  --likelihood_format logits \
  --samples_between_prediction 20 \
  --maj_vote_range future \
  --saved_checkpoint_pth <path_to_your_pth_checkpoint> \
  --epn_eval 0 \
  --verbose 1 \
  --model_choice any2any
```
To remove all smoothing, set `--stride 20`, `--lookahead 0`, `--samples_between_prediction 1`, and `--maj_vote_range single`.

To evaluate EPN-only models, set `--files_or_dirs ../data/EMG-EPN-612` and `--epn_eval 1`.

### Output

The evaluation code produces three outputs under `output/`:
- Summary txt: Overall raw & transition accuracy (mean ± std), event counts, and a tally of failure reasons.
- Per-file JSON: Metrics plus full ground-truth & prediction sequences for each file.
- PNG plots: 3-panel figure: 8-channel EMG, ground-truth labels, and model predictions over time.

---

## :hospital: Stroke Experiments (Healthy-to-Stroke Transfer Learning)

This repository includes experimental infrastructure for studying healthy-to-stroke few-shot adaptation for sEMG intent detection. The experiments evaluate different fine-tuning strategies when adapting a healthy-pretrained model to stroke participants.

### Experimental Setup

**Participants**: p15, p20 (both left-hand stroke survivors)

**Data Organization**:
- Calibration pool: 4 baseline sets (open_1-4, close_1-4)
- Test conditions: mid-session baseline, end-session fatigue, unseen posture, sensor shift, orthosis-actuated

**Fine-tuning Strategies**:
1. Zero-shot (frozen pretrained model)
2. Stroke-only (train from scratch)
3. Head-only (freeze backbone, train classification head only)
4. LoRA (low-rank adaptation)
5. Full fine-tuning (update all parameters)

### Running the Experiments

#### 1. Validation Tests

First, verify all implementations work correctly:

```bash
cd reactemg/
python3 test_implementations.py
```

This tests repetition extraction, nested sampling, and verifies all scripts exist.

#### 2. Main Experiment (Complete Pipeline)

Run the full main experiment for all participants and variants:

```bash
cd reactemg/
python3 run_main_experiment.py
```

This will:
- Evaluate pretrained model zero-shot on stroke data
- Run 4-fold CV hyperparameter search for each variant (36 configs × 4 folds)
- Train final models with best hyperparameters on full calibration pool
- Evaluate all models on 5 test conditions
- Save results to `results/main_experiment/`

**Expected runtime**: ~40-100 GPU hours (1,152 training runs)

#### 3. Hyperparameter Search (Individual Variant)

To run CV search for a specific variant only:

```bash
cd reactemg/
python3 cv_hyperparameter_search.py \
  --participant p15 \
  --participant_folder ~/Workspace/myhand/src/collected_data/2025_12_04 \
  --variant lora \
  --pretrained_checkpoint /path/to/pretrained.pth
```

Variants: `stroke_only`, `head_only`, `lora`, `full_finetune`

Results saved to: `temp_cv_checkpoints/p15_lora_cv_results.json`

#### 4. Data Efficiency Experiment

After identifying the best variant from main experiment:

```bash
cd reactemg/
python3 run_data_efficiency.py \
  --participant p15 \
  --variant lora \
  --config_file temp_cv_checkpoints/p15_lora_cv_results.json
```

This evaluates performance with K=1,4,8 paired repetitions (12 trials per K).

Results saved to: `results/data_efficiency/p15/`

#### 5. Convergence Study

Track convergence and potential catastrophic forgetting:

```bash
cd reactemg/
python3 run_convergence.py \
  --participant p15 \
  --variant lora \
  --config_file temp_cv_checkpoints/p15_lora_cv_results.json
```

This trains for 10× the optimal epochs and evaluates each epoch on both stroke test sets and healthy s15 data.

Results saved to: `results/convergence/p15/`

### Fine-Tuning Strategies

#### Head-Only Fine-Tuning

Freeze all parameters except the action prediction head:

```bash
python3 main.py \
  --freeze_backbone 1 \
  --saved_checkpoint_pth /path/to/pretrained.pth \
  [other training args]
```

#### LoRA Fine-Tuning

Low-rank adaptation with fixed hyperparameters (rank=16, alpha=8):

```bash
python3 main.py \
  --use_lora 1 \
  --lora_rank 16 \
  --lora_alpha 8 \
  --lora_dropout_p 0.05 \
  --saved_checkpoint_pth /path/to/pretrained.pth \
  [other training args]
```

#### Full Fine-Tuning

Update all parameters:

```bash
python3 main.py \
  --saved_checkpoint_pth /path/to/pretrained.pth \
  [other training args]
```

#### Train from Scratch (Stroke-Only)

Omit the pretrained checkpoint:

```bash
python3 main.py \
  [training args without --saved_checkpoint_pth]
```

### Hyperparameter Search Space

**Searched parameters** (36 combinations):
- Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
- Epochs: [5, 10, 15]
- Dropout: [0.1, 0.3, 0.5]

**Fixed parameters**:
- Batch size: 128
- Window size: 600
- LoRA rank: 16
- LoRA alpha: 8
- LoRA dropout: 0.05

**Selection criteria**:
- Primary: Highest average transition accuracy across 4 CV folds
- Tiebreaker: Fewer epochs

### Evaluation Parameters

All evaluations use consistent parameters:
```bash
--buffer_range 800
--lookahead 100
--samples_between_prediction 100
--allow_relax 1
--stride 1
--likelihood_format logits
--maj_vote_range future
```

### Output Structure

```
results/
├── main_experiment/
│   ├── p15/
│   │   ├── zero_shot/
│   │   │   ├── mid_session_baseline/
│   │   │   ├── end_session_baseline/
│   │   │   ├── unseen_posture/
│   │   │   ├── sensor_shift/
│   │   │   └── orthosis_actuated/
│   │   ├── stroke_only/
│   │   ├── head_only/
│   │   ├── lora/
│   │   └── full_finetune/
│   └── p20/ [same structure]
│
├── data_efficiency/
│   ├── p15/
│   │   ├── K1/ [12 trials + aggregated_metrics.json]
│   │   ├── K4/
│   │   └── K8/
│   └── p20/
│
└── convergence/
    ├── p15/
    │   ├── frozen_baseline/
    │   ├── epoch_1/ ... epoch_N/
    │   └── convergence_curves.json
    └── p20/

model_checkpoints/
├── main_experiment/
│   ├── p15_stroke_only_final.pth
│   ├── p15_head_only_final.pth
│   ├── p15_lora_final.pth
│   └── p15_full_finetune_final.pth
├── data_efficiency/
│   └── p15/ [K1_trial0.pth ... K8_trial11.pth]
└── convergence/
    └── p15/ [epoch_1.pth ... epoch_N.pth]
```

### Programmatic Evaluation

For custom evaluation scripts, use the programmatic API:

```python
from event_classification import evaluate_checkpoint_programmatic

metrics = evaluate_checkpoint_programmatic(
    checkpoint_path='model.pth',
    csv_files=['p15_open_1.csv', 'p15_close_1.csv'],
    buffer_range=800,
    lookahead=100,
    samples_between_prediction=100,
    allow_relax=1,
    stride=1,
    model_choice="any2any",
    verbose=0,
)

print(f"Transition Accuracy: {metrics['transition_accuracy']:.4f}")
print(f"Raw Accuracy: {metrics['raw_accuracy']:.4f}")
```

### Key Implementation Files

**Core modifications**:
- `main.py`: Added `--freeze_backbone` flag for head-only fine-tuning
- `dataset.py`: Added `sampled_segments` parameter for data efficiency experiments
- `event_classification.py`: Added `evaluate_checkpoint_programmatic()` function

**New utilities**:
- `dataset_utils.py`: Repetition extraction and nested sampling functions
- `cv_hyperparameter_search.py`: 4-fold CV with automated hyperparameter selection
- `run_main_experiment.py`: Complete main experiment orchestration
- `run_data_efficiency.py`: Data efficiency experiment (K=1,4,8)
- `run_convergence.py`: Convergence and catastrophic forgetting tracking
- `test_implementations.py`: Validation test suite

### Troubleshooting

**"No checkpoint found"**:
- Verify training completed successfully
- Check `model_checkpoints/` directory exists
- Ensure epoch number matches configuration

**"Test data not found"**:
- Verify participant data paths in script configuration
- Check file naming patterns match (e.g., `p15_open_1.csv`)
- Ensure all baseline and test files exist

**CUDA out of memory**:
- Reduce batch size (default: 128)
- Run experiments sequentially instead of parallel
- Use gradient accumulation (requires code modification)

---

## :memo: Citation
If you find this codebase useful, consider citing:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2506.19815,
  doi = {10.48550/ARXIV.2506.19815},
  url = {https://arxiv.org/abs/2506.19815},
  author = {Wang,  Runsheng and Zhu,  Xinyue and Chen,  Ava and Xu,  Jingxi and Winterbottom,  Lauren and Nilsen,  Dawn M. and Stein,  Joel and Ciocarlie,  Matei},
  keywords = {Robotics (cs.RO),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {ReactEMG: Zero-Shot,  Low-Latency Intent Detection via sEMG},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## :email: Contact
For questions or support, please email Runsheng at runsheng.w@columbia.edu

## :scroll: License
This project is released under the MIT License; see the [License](LICENSE) file for full details.

## :handshake: Acknowledgments
This work was supported in part by an Amazon Research Award and the Columbia University Data Science Institute Seed Program. Ava Chen was supported by NIH grant 1F31HD111301-01A1. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the sponsors. We would like to thank Katelyn Lee, Eugene Sohn, Do-Gon Kim, and Dilara Baysal for their assistance with the hand orthosis hardware. We thank Zhanpeng He and Gagan Khandate for their helpful feedback and insightful discussions.

