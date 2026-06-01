import torch
import random
import pandas as pd
import numpy as np
import scipy.signal
from torch.utils.data import Dataset
from scipy.signal import medfilt
from tqdm import tqdm
import math
from typing import List, Optional, Dict, Tuple
from torch import nn
import torch.nn.functional as F


#################################################################
######################## Any2Any Dataset ########################
#################################################################
class Any2Any_Dataset(Dataset):
    def __init__(
        self,
        labeled_csv_paths,
        unlabeled_csv_paths,
        median_filter_size,
        window_size,
        offset,
        embedding_method,
        lambda_poisson,
        seeded_mask,
        sampling_probability_poisson,
        poisson_mask_percentage_sampling_range,
        end_mask_percentage_sampling_range,
        task_selection,
        stage_1_weights,
        stage_2_weights,
        mask_alignment,
        transition_buffer,
        mask_tokens_dict,
        with_training_curriculum,
        num_classes,
        medfilt_order,
        noise,
        hand_choice,
        eval_mode=False,
        eval_task=None,
        transition_samples_only=False,
        mask_percentage=0.6,
        mask_type="poisson",
        sampled_segments=None,
    ):
        """
        This dataset supports both labeled and unlabeled data in csv/numpy format, applies
        median filtering and rectification,
        and prepares data for various tasks (predict action, predict EMG, etc.) with masking.

        The dataset can operate in training mode (with curriculum learning over multiple stages)
        or evaluation mode (with a fixed masking scheme). It also handles transition samples.

        This dataset is used for both training and simulated online inference

        Args:
            labeled_csv_paths (List[str]):
                Paths to labeled CSV/NPY files containing EMG data and action labels.
            unlabeled_csv_paths (List[str]):
                Paths to unlabeled CSV/NPY files containing EMG data.
            median_filter_size (int):
                Kernel size for median filtering of EMG signals.
            window_size (int):
                Number of timesteps per sample (outer window size).
            offset (int):
                Step size to slide over the sequence when creating samples.
            embedding_method (str):
                Method for representing EMG/action tokens (e.g. "linear_projection").
            lambda_poisson (float):
                Lambda parameter for Poisson-based masking.
            seeded_mask (bool):
                If True, uses a seeded (deterministic) approach to generate masks.
            sampling_probability_poisson (float):
                Probability of selecting Poisson masking vs. other types (e.g., "end").
            poisson_mask_percentage_sampling_range (Dict[int, Tuple[float, float]]):
                Per-task range for sampling the percentage of Poisson-mask coverage.
            end_mask_percentage_sampling_range (Dict[int, Tuple[float, float]]):
                Per-task range for sampling the percentage of end-based mask coverage.
            task_selection (List[int]):
                List of task indices. Tasks map to:
                0 = Predict action from EMG (dense labeling),
                1 = Predict EMG from action,
                2 = Bidirectional (mask both EMG and action),
                3 = Unlabeled/self-supervised EMG.
            stage_1_weights (List[float]):
                Weights used for sampling stage 0 vs. stage 1 in the curriculum.
            stage_2_weights (List[float]):
                Weights used for sampling stage 0 vs. stage 2 in the curriculum.
            mask_alignment (str):
                Mask alignment strategy: "aligned" or "non-aligned" across channels.
            transition_buffer (int):
                Number of timesteps around the transition index for targeted masking.
            mask_tokens_dict (Dict[str, Dict[str, float]]):
                Dictionary specifying mask token values per embedding method (e.g. for EMG, action).
            with_training_curriculum (bool):
                If True, uses a curriculum learning approach over multiple stages.
            num_classes (int):
                Number of action classes (used for classification).
            medfilt_order (str):
                Whether median filtering is done before or after rectification ("before_rec" or "after_rec").
            noise (float):
                Amplitude of uniform noise added to EMG signals (0.0 for no noise).
            hand_choice (str):
                Which hand's EMG data is being used, "left" or "right" (remaps channels for "left").
            eval_mode (bool, optional):
                If True, dataset is used for evaluation (fixed masking). Defaults to False.
            eval_task (str, optional):
                Task for evaluation mode. One of {"predict_action", "predict_emg", "predict_emg_ss"}.
                Defaults to None.
            transition_samples_only (bool, optional):
                If True, filters labeled data to only keep samples containing transitions.
                Defaults to False.
            mask_percentage (float, optional):
                Percentage of time steps to mask during evaluation. Defaults to 0.6.
            mask_type (str, optional):
                Masking strategy for evaluation. One of {"poisson", "end", "targeted"}.
                Defaults to "poisson".

        Attributes:
            all_data (List[Tuple[np.ndarray, np.ndarray, int]]):
                Master list of all loaded samples (EMG, action, transition_index).
            raw_labeled_data (List[Tuple[np.ndarray, np.ndarray, int]]):
                Loaded labeled samples (before augmentation/masking).
            noisy_labeled_data (List[Any]):
                Not currently used (placeholder for future expansions).
            augmented_transition_samples (List[Any]):
                Placeholder list for additional transition-augmented samples.
            raw_unlabeled_data (List[Any]):
                Loaded unlabeled samples (placeholder for future expansions).
            noisy_unlabeled_data (List[Any]):
                Placeholder list for unlabeled noisy samples.
            untokenized_data (List[Tuple[np.ndarray, np.ndarray, int]]):
                Stores unmasked data for evaluation/analysis.

        Raises:
            Exception: If the CSV does not contain a "gt" column for action labels.
            ValueError: If file extension is not recognized, or if `medfilt_order`
                or `mask_type` is invalid.
        """

        self.all_data = []
        self.raw_labeled_data = []
        self.noisy_labeled_data = []
        self.augmented_transition_samples = []
        self.raw_unlabeled_data = []
        self.noisy_unlabeled_data = []
        self.window_size = window_size
        self.embedding_method = embedding_method
        self.lambda_poisson = lambda_poisson
        self.sampling_probability_poisson = sampling_probability_poisson
        self.poisson_mask_percentage_sampling_range = (
            poisson_mask_percentage_sampling_range
        )
        self.end_mask_percentage_sampling_range = end_mask_percentage_sampling_range
        self.seeded_mask = seeded_mask
        self.task_selection = task_selection
        self.mask_alignment = mask_alignment
        self.transition_buffer = transition_buffer
        self.mask_tokens_dict = mask_tokens_dict
        self.num_classes = num_classes
        self.noise_amplitude = noise
        self.hand_choice = hand_choice

        self.curriculum_stage = 0
        self.stage_1_weights = stage_1_weights
        self.stage_2_weights = stage_2_weights
        self.with_training_curriculum = with_training_curriculum
        self.class_data = {0: 0, 1: 0, 2: 0}

        self.cur_epoch = 0

        self.eval_mode = eval_mode
        self.eval_task = eval_task
        self.transition_samples_only = transition_samples_only
        self.eval_mask_percentage = mask_percentage
        self.eval_mask_type = mask_type
        self.untokenized_data = []
        self.sampled_segments = sampled_segments  # Dict: {file_path: [(start, end), ...]}

        if not self.eval_mode:
            if any(item in [0, 1, 2] for item in self.task_selection):
                for path in tqdm(labeled_csv_paths):
                    temp_raw_labeled_data, temp_noisy_labeled_data = (
                        self.multivariate_preprocessing(
                            path,
                            median_filter_size,
                            window_size,
                            offset,
                            embedding_method,
                            medfilt_order,
                            hand_choice,
                        )
                    )
                    self.raw_labeled_data.extend(temp_raw_labeled_data)

                self.all_data.extend(self.raw_labeled_data)
                self.all_data.extend(self.noisy_labeled_data)
                self.all_data.extend(self.augmented_transition_samples)

            self.modality_switch_index = len(self.all_data)

            if any(item in [3] for item in self.task_selection):
                for path in tqdm(unlabeled_csv_paths):
                    temp_raw_unlabeled_data, temp_noisy_unlabeled_data = (
                        self.multivariate_preprocessing(
                            path,
                            median_filter_size,
                            window_size,
                            offset,
                            embedding_method,
                            medfilt_order,
                            hand_choice,
                        )
                    )
                    self.all_data.extend(temp_raw_unlabeled_data)
                    self.all_data.extend(temp_noisy_unlabeled_data)

            if 3 not in self.task_selection:
                self.len_multiplier = len(self.task_selection)
            else:
                self.len_multiplier = len(self.task_selection) - 1
            self.num_labeled_samples = self.modality_switch_index * self.len_multiplier
            self.num_unlabeled_samples = len(self.all_data) - self.modality_switch_index
            self.labeled_indices = list(range(self.num_labeled_samples))
            self.unlabeled_indices = list(
                range(
                    self.num_labeled_samples,
                    self.num_labeled_samples + self.num_unlabeled_samples,
                )
            )

        else:
            for path in tqdm(labeled_csv_paths):
                temp_raw_labeled_data, _ = self.multivariate_preprocessing(
                    path,
                    median_filter_size,
                    window_size,
                    offset,
                    embedding_method,
                    medfilt_order,
                    hand_choice,
                )
                temp_raw_labeled_data_untokenized, _ = self.multivariate_preprocessing(
                    path,
                    median_filter_size,
                    window_size,
                    offset,
                    embedding_method,
                    medfilt_order,
                    hand_choice,
                )

                if self.transition_samples_only:
                    filtered_data = []
                    filtered_untok = []
                    for sample_idx in range(len(temp_raw_labeled_data)):
                        (
                            _emg,
                            _act,
                            _tindex,
                        ) = temp_raw_labeled_data[sample_idx]
                        if _tindex != -1:
                            filtered_data.append(temp_raw_labeled_data[sample_idx])
                            filtered_untok.append(
                                temp_raw_labeled_data_untokenized[sample_idx]
                            )
                    temp_raw_labeled_data = filtered_data
                    temp_raw_labeled_data_untokenized = filtered_untok

                self.all_data.extend(temp_raw_labeled_data)
                self.untokenized_data.extend(temp_raw_labeled_data_untokenized)

            self.modality_switch_index = 0
            self.num_labeled_samples = len(self.all_data)
            self.num_unlabeled_samples = 0
            self.labeled_indices = list(range(self.num_labeled_samples))
            self.unlabeled_indices = []

    def multivariate_preprocessing(
        self,
        path,
        median_filter_size,
        window_size,
        offset,
        embedding_method,
        medfilt_order,
        hand_choice,
    ):
        extracted_samples = []
        extracted_unlabeled_samples = []

        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
            if "gt" not in df.columns:
                raise Exception("gt column not found")
            action_sequence = df["gt"].to_numpy()
            try:
                df_emg = df[
                    [
                        "emg_0",
                        "emg_1",
                        "emg_2",
                        "emg_3",
                        "emg_4",
                        "emg_5",
                        "emg_6",
                        "emg_7",
                    ]
                ]
            except KeyError:
                df_emg = df[
                    ["emg0", "emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7"]
                ]
            if hand_choice == "left":
                remap_order = [6, 5, 4, 3, 2, 1, 0, 7]
                data_array = df_emg.to_numpy().astype(np.int16)
                data_array = data_array[:, remap_order]
                df_emg = pd.DataFrame(
                    data_array, columns=["emg_" + str(i) for i in range(8)]
                )

            if medfilt_order == "before_rec":
                filtered_data = df_emg.apply(
                    lambda x: medfilt(x, kernel_size=median_filter_size)
                )
                rectified_data = np.abs(filtered_data)
            elif medfilt_order == "after_rec":
                rectified_data = np.abs(df_emg)
                rectified_data = rectified_data.apply(
                    lambda x: medfilt(x, kernel_size=median_filter_size)
                )
            else:
                raise ValueError(
                    "medfilt_order must be either 'before_rec' or 'after_rec'"
                )
            scaled_data = rectified_data / 128.0

        elif path.lower().endswith(".npy"):
            loaded = np.load(path).astype(np.float32)
            action_sequence = loaded[:, 0]
            data_array = loaded[:, 1:]
            if hand_choice == "left":
                remap_order = [6, 5, 4, 3, 2, 1, 0, 7]
                data_array = data_array[:, remap_order]

            if medfilt_order == "before_rec":
                for i in range(data_array.shape[1]):
                    data_array[:, i] = medfilt(
                        data_array[:, i], kernel_size=median_filter_size
                    )
                data_array = np.abs(data_array)
            elif medfilt_order == "after_rec":
                data_array = np.abs(data_array)
                for i in range(data_array.shape[1]):
                    data_array[:, i] = medfilt(
                        data_array[:, i], kernel_size=median_filter_size
                    )
            else:
                raise ValueError(
                    "medfilt_order must be either 'before_rec' or 'after_rec'"
                )
            scaled_data = data_array / 128.0
        else:
            raise ValueError("File extension not recognized. Must be .csv or .npy")

        if isinstance(scaled_data, pd.DataFrame):
            clipped_data = scaled_data.to_numpy().astype(np.float32)
        else:
            clipped_data = scaled_data.astype(np.float32)

        # Handle sampled segments: extract only specified segments from this file
        if self.sampled_segments is not None and path in self.sampled_segments:
            segment_indices = self.sampled_segments[path]
            # Extract and concatenate specified segments
            segment_data_list = []
            segment_action_list = []
            for start_idx, end_idx in segment_indices:
                segment_data_list.append(clipped_data[start_idx:end_idx])
                segment_action_list.append(action_sequence[start_idx:end_idx])
            clipped_data = np.vstack(segment_data_list)
            action_sequence = np.concatenate(segment_action_list)

        for start in range(0, clipped_data.shape[0] - window_size + 1, offset):
            window = clipped_data[start : start + window_size, :]
            windowed_action_sequence = action_sequence[start : start + window_size]

            transition_list = np.where(
                windowed_action_sequence[:-1] != windowed_action_sequence[1:]
            )[0]
            transition_index = (
                transition_list[0] if len(transition_list) > 0 else -1
            )

            extracted_samples.append(
                (
                    window.astype(np.float32),
                    windowed_action_sequence.astype(np.int64),
                    transition_index,
                )
            )

        return extracted_samples, extracted_unlabeled_samples

    def masking_a2a(
        self,
        sequence,
        mask_percentage,
        mask_type,
        mask_channel_selection,
        mask_alignment,
        lambda_poisson,
        seeded_mask,
        mask_token,
        transition_index,
        transition_buffer,
        use_bert_mask,
    ):
        univariate_sequence_status = False
        if sequence.ndim == 1:
            univariate_sequence_status = True
            sequence = sequence.reshape(-1, 1)

        window_size, num_channel = sequence.shape

        if mask_type not in ["poisson", "end", "targeted"]:
            raise ValueError("mask_type must be 'poisson', 'end', or 'targeted'")

        if any(channel >= num_channel for channel in mask_channel_selection):
            raise ValueError("mask_channel_selection contains invalid channel indices")

        total_tokens_to_mask = int(window_size * mask_percentage)
        mask_positions = np.zeros_like(sequence, dtype=bool)
        rng = np.random if seeded_mask else np.random.default_rng()

        if mask_type == "poisson":
            if mask_alignment == "aligned":
                tokens_masked = 0
                while tokens_masked < total_tokens_to_mask:
                    span_length = rng.poisson(lambda_poisson)
                    if span_length <= 0 or span_length >= window_size:
                        continue
                    if tokens_masked + span_length > total_tokens_to_mask:
                        span_length = total_tokens_to_mask - tokens_masked
                    start_pos = (
                        rng.randint(0, max(1, window_size - span_length + 1))
                        if seeded_mask
                        else rng.integers(0, max(1, window_size - span_length + 1))
                    )
                    end_pos = start_pos + span_length
                    already_masked = mask_positions[
                        start_pos:end_pos, mask_channel_selection[0]
                    ]
                    positions_not_already_masked_indices = np.where(~already_masked)[0]
                    num_new_positions = len(positions_not_already_masked_indices)
                    if num_new_positions == 0:
                        continue
                    if tokens_masked + num_new_positions > total_tokens_to_mask:
                        num_needed = total_tokens_to_mask - tokens_masked
                        positions_not_already_masked_indices = (
                            positions_not_already_masked_indices[:num_needed]
                        )
                        num_new_positions = len(positions_not_already_masked_indices)
                    for channel in mask_channel_selection:
                        mask_positions[start_pos:end_pos, channel][
                            positions_not_already_masked_indices
                        ] = True
                    tokens_masked += num_new_positions

            elif mask_alignment == "non-aligned":
                for channel in mask_channel_selection:
                    tokens_masked = 0
                    while tokens_masked < total_tokens_to_mask:
                        span_length = rng.poisson(lambda_poisson)
                        if span_length <= 0 or span_length >= window_size:
                            continue
                        if tokens_masked + span_length > total_tokens_to_mask:
                            span_length = total_tokens_to_mask - tokens_masked
                        start_pos = (
                            rng.randint(0, max(1, window_size - span_length + 1))
                            if seeded_mask
                            else rng.integers(0, max(1, window_size - span_length + 1))
                        )
                        end_pos = start_pos + span_length
                        already_masked = mask_positions[start_pos:end_pos, channel]
                        positions_not_already_masked_indices = np.where(
                            ~already_masked
                        )[0]
                        num_new_positions = len(positions_not_already_masked_indices)
                        if num_new_positions == 0:
                            continue
                        if tokens_masked + num_new_positions > total_tokens_to_mask:
                            num_needed = total_tokens_to_mask - tokens_masked
                            positions_not_already_masked_indices = (
                                positions_not_already_masked_indices[:num_needed]
                            )
                            num_new_positions = len(
                                positions_not_already_masked_indices
                            )
                        mask_positions[start_pos:end_pos, channel][
                            positions_not_already_masked_indices
                        ] = True
                        tokens_masked += num_new_positions

        elif mask_type == "end":
            start_pos = window_size - total_tokens_to_mask
            for channel in mask_channel_selection:
                mask_positions[start_pos:, channel] = True

        elif mask_type == "targeted":
            for channel in mask_channel_selection:
                mask_positions[
                    transition_index - transition_buffer : transition_index
                    + transition_buffer,
                    channel,
                ] = True

        masked_sequence = np.where(mask_positions, mask_token, sequence)
        if univariate_sequence_status:
            masked_sequence = masked_sequence.squeeze()
            mask_positions = mask_positions.squeeze()
        return masked_sequence, mask_positions

    def __len__(self):
        if not self.eval_mode:
            if self.num_unlabeled_samples == 0:
                return self.num_labeled_samples
            else:
                return self.num_labeled_samples + self.num_unlabeled_samples
        else:
            return len(self.all_data)

    def __getitem__(self, idx):
        if not self.eval_mode:
            if idx < self.num_labeled_samples:
                raw_idx = idx // self.len_multiplier
                task_idx_unmapped = idx % self.len_multiplier
                task_idx = self.task_selection[task_idx_unmapped]
            else:
                raw_idx = self.modality_switch_index + (idx - self.num_labeled_samples)
                task_idx = 3

            emg_window, action_window, transition_index = self.all_data[raw_idx]

            if self.noise_amplitude > 0.0:
                amplitude = self.noise_amplitude
                noise = np.random.uniform(-amplitude, amplitude, size=emg_window.shape)
                emg_window = emg_window + noise
                emg_window = np.clip(emg_window, 0.0, 1.0)
            emg_window = emg_window.astype(np.float32)

            mask_type = (
                "poisson"
                if random.random() < self.sampling_probability_poisson
                else ("poisson" if task_idx == 3 else "end")
            )
            if mask_type == "poisson":
                mask_lower_bound, mask_upper_bound = (
                    self.poisson_mask_percentage_sampling_range[task_idx]
                )
            else:
                mask_lower_bound, mask_upper_bound = (
                    self.end_mask_percentage_sampling_range[task_idx]
                )

            if task_idx == 3:
                mask_percentage = random.uniform(
                    mask_lower_bound - 0.1, mask_upper_bound - 0.1
                )
            else:
                mask_percentage = random.uniform(mask_lower_bound, mask_upper_bound)

            selected_curriculum_stage = [0]
            if self.curriculum_stage == 1 and task_idx in [1, 3]:
                selected_curriculum_stage = random.choices([0, 1], self.stage_1_weights)
            if self.curriculum_stage == 2 and task_idx == 0:
                if (
                    transition_index >= self.transition_buffer
                    and transition_index <= self.window_size - self.transition_buffer
                ):
                    selected_curriculum_stage = [2]
                else:
                    selected_curriculum_stage = random.choices(
                        [0, 2], self.stage_2_weights
                    )

            action_mask_channel_selection = [0]
            use_bert_mask = False
            if selected_curriculum_stage == [0]:
                emg_mask_channel_selection = list(range(emg_window.shape[1]))
                use_bert_mask = True
            elif selected_curriculum_stage == [1]:
                mask_type = "end"
                mask_percentage = 1.0
                if self.embedding_method in ["linear_projection", "separate_channel"]:
                    emg_mask_channel_selection = random.sample(
                        list(range(emg_window.shape[1])), 2
                    )
                else:
                    raise Exception("unsupported embedding_method")
            elif selected_curriculum_stage == [2]:
                if (
                    transition_index > self.transition_buffer
                    and transition_index < self.window_size - self.transition_buffer
                ):
                    if random.choice([0, 1]) == 1:
                        mask_type = "targeted"
                    else:
                        mask_type = "end"
                        mask_percentage = 1.0
                else:
                    mask_type = "end"
                    mask_percentage = 1.0
            else:
                raise Exception("selected_curriculum_stage not recognized")

            emg_dummy_mask_positions = np.zeros_like(emg_window, dtype=bool)
            action_dummy_mask_positions = np.zeros_like(action_window, dtype=bool)
            emg_mask_token = self.mask_tokens_dict[self.embedding_method]["EMG_mask"]
            action_mask_token = self.mask_tokens_dict[self.embedding_method][
                "Action_mask"
            ]

            if task_idx == 0:
                masked_actions, mask_positions_actions = self.masking_a2a(
                    action_window,
                    mask_percentage,
                    mask_type,
                    action_mask_channel_selection,
                    self.mask_alignment,
                    self.lambda_poisson,
                    self.seeded_mask,
                    action_mask_token,
                    transition_index,
                    self.transition_buffer,
                    use_bert_mask,
                )
                masked_emg = emg_window
                mask_positions_emg = emg_dummy_mask_positions
            elif task_idx == 1:
                masked_emg, mask_positions_emg = self.masking_a2a(
                    emg_window,
                    mask_percentage,
                    mask_type,
                    emg_mask_channel_selection,
                    self.mask_alignment,
                    self.lambda_poisson,
                    self.seeded_mask,
                    emg_mask_token,
                    transition_index,
                    self.transition_buffer,
                    use_bert_mask,
                )
                masked_actions = action_window
                mask_positions_actions = action_dummy_mask_positions
            elif task_idx == 2:
                masked_actions, mask_positions_actions = self.masking_a2a(
                    action_window,
                    mask_percentage,
                    mask_type,
                    action_mask_channel_selection,
                    self.mask_alignment,
                    self.lambda_poisson,
                    self.seeded_mask,
                    action_mask_token,
                    transition_index,
                    self.transition_buffer,
                    use_bert_mask,
                )
                masked_emg, mask_positions_emg = self.masking_a2a(
                    emg_window,
                    mask_percentage,
                    mask_type,
                    emg_mask_channel_selection,
                    self.mask_alignment,
                    self.lambda_poisson,
                    self.seeded_mask,
                    emg_mask_token,
                    transition_index,
                    self.transition_buffer,
                    use_bert_mask,
                )
            elif task_idx == 3:
                masked_emg, mask_positions_emg = self.masking_a2a(
                    emg_window,
                    mask_percentage,
                    mask_type,
                    emg_mask_channel_selection,
                    self.mask_alignment,
                    self.lambda_poisson,
                    self.seeded_mask,
                    emg_mask_token,
                    transition_index,
                    self.transition_buffer,
                    use_bert_mask,
                )
                masked_actions = np.full(action_window.shape, action_mask_token)
                mask_positions_actions = action_dummy_mask_positions
            else:
                raise Exception("task_idx not recognized")

            return (
                emg_window,
                action_window,
                masked_emg,
                masked_actions,
                mask_positions_emg,
                mask_positions_actions,
                task_idx,
                transition_index,
            )
        else:
            # Inference
            (emg_window, action_window, transition_index) = self.all_data[idx]
            untokenized_emg = self.untokenized_data[idx][0]

            if self.eval_task == "predict_action":
                task_idx = 0
            elif self.eval_task == "predict_emg":
                task_idx = 1
            elif self.eval_task == "predict_emg_ss":
                task_idx = 3
            else:
                task_idx = 0

            emg_dummy_mask_positions = np.zeros_like(emg_window, dtype=bool)
            action_dummy_mask_positions = np.zeros_like(action_window, dtype=bool)
            emg_mask_token = self.mask_tokens_dict[self.embedding_method]["EMG_mask"]
            action_mask_token = self.mask_tokens_dict[self.embedding_method][
                "Action_mask"
            ]
            use_bert_mask = False

            if self.eval_task == "predict_action":
                masked_actions, mask_positions_actions = self.masking_a2a(
                    action_window,
                    1.0,
                    "end",
                    [0],
                    "non-aligned",
                    1,
                    self.seeded_mask,
                    action_mask_token,
                    transition_index,
                    self.transition_buffer,
                    use_bert_mask,
                )
                masked_emg = emg_window
                mask_positions_emg = emg_dummy_mask_positions
            elif self.eval_task == "predict_emg":
                if self.embedding_method == "linear_projection":
                    masked_emg, mask_positions_emg = self.masking_a2a(
                        emg_window,
                        self.eval_mask_percentage,
                        self.eval_mask_type,
                        range(emg_window.shape[1]),
                        "aligned",
                        1,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        self.transition_buffer,
                        use_bert_mask,
                    )
                    masked_actions = action_window
                    mask_positions_actions = action_dummy_mask_positions
                else:
                    raise Exception(
                        "eval_task='predict_emg' but embedding_method not recognized"
                    )
            else:
                raise Exception(
                    f"eval_task {self.eval_task} not recognized in inference."
                )

            return (
                emg_window,
                action_window,
                masked_emg,
                masked_actions,
                mask_positions_emg,
                mask_positions_actions,
                task_idx,
                transition_index,
                untokenized_emg,
            )


############################################################
