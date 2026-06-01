import os, socket, datetime, math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import numpy as np


###################################
############# Any2Any #############
###################################
class Any2Any_Model(nn.Module):
    def __init__(
        self,
        embedding_dim,
        nhead,
        dropout,
        activation,
        num_layers,
        window_size,
        embedding_method,
        mask_alignment,
        share_pe,
        tie_weight,
        use_input_layernorm,
        num_classes,
        output_reduction_method,
        chunk_size,
    ):
        super(Any2Any_Model, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        self.nhead = nhead
        self.mask_alignment = mask_alignment
        self.use_input_layernorm = use_input_layernorm
        self.output_reduction_method = output_reduction_method
        self.chunk_size = chunk_size
        self.window_size = window_size

        # Action vocab + embedding
        self.action_vocab_size = num_classes + 1  # same as before
        self.action_embedding = nn.Embedding(self.action_vocab_size, embedding_dim)

        # Modality-specific embedding for actions
        self.action_modality_specific_embedding = nn.Parameter(
            torch.empty(1, 1, embedding_dim)
        )
        nn.init.uniform_(self.action_modality_specific_embedding, a=-0.02, b=0.02)

        # Transformer encoder definition
        # Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Action output projection
        self.action_output_projection = nn.Linear(embedding_dim, self.action_vocab_size)

        self.emg_embedding = nn.Conv1d(8, embedding_dim, kernel_size=1, stride=1)
        self.linear_projection_learnable_mask = nn.Parameter(
            torch.empty(1, embedding_dim, 1)
        )
        nn.init.uniform_(self.linear_projection_learnable_mask, a=-0.02, b=0.02)

        if self.use_input_layernorm:
            self.emg_in_layer_norm = nn.LayerNorm(embedding_dim)

        # Modality-specific embedding for EMG
        self.emg_modality_specific_embedding = nn.ParameterDict()
        self.emg_modality_specific_embedding["emg_id_embedding"] = nn.Parameter(
            torch.empty(1, 1, embedding_dim)
        )
        nn.init.uniform_(
            self.emg_modality_specific_embedding["emg_id_embedding"],
            a=-0.02,
            b=0.02,
        )

        # Positional encoding of size window_size
        if share_pe:
            self.emg_positional_encoding = nn.Parameter(
                torch.empty(1, self.window_size, embedding_dim)
            )
            nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
            self.action_positional_encoding = self.emg_positional_encoding
        else:
            self.emg_positional_encoding = nn.Parameter(
                torch.empty(1, self.window_size, embedding_dim)
            )
            self.action_positional_encoding = nn.Parameter(
                torch.empty(1, self.window_size, embedding_dim)
            )
            nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
            nn.init.uniform_(self.action_positional_encoding, a=-0.02, b=0.02)

        # EMG output projection
        self.emg_output_projection = nn.Conv1d(
            embedding_dim, 8, kernel_size=1, stride=1
        )

        # Tie action embedding with action output projection if specified
        if tie_weight:
            self.action_output_projection.weight = self.action_embedding.weight
            pass

        # Possibly add aggregator for chunk-based output, as in
        if self.output_reduction_method == "learned":
            self.chunk_aggregator = nn.Linear(
                self.chunk_size * embedding_dim, embedding_dim
            )
        else:
            self.chunk_aggregator = None

    def _reduce_sequence_by_pooling(self, x, chunk_size):
        """
        x: (B, T, d_model)
        Pool in time dimension with chunk_size
        """
        B, T, D = x.size()
        # Assume T % chunk_size == 0
        x = x.reshape(B, T // chunk_size, chunk_size, D)  # (B, T//cs, cs, D)
        return x.mean(dim=2)  # (B, T//cs, D)

    def _reduce_sequence_by_learned(self, x, chunk_size, aggregator):
        """
        x: (B, T, d_model)
        Flatten each chunk, then apply aggregator
        """
        B, T, D = x.size()
        x = x.reshape(B, T // chunk_size, chunk_size * D)  # (B, T//cs, cs*D)
        return aggregator(x)  # (B, T//cs, d_model)

    def forward(
        self,
        masked_emg,
        masked_actions,
        task_idx,
        mask_positions_emg,
        return_output=False,
        emg_window=None,
        action_window=None,
    ):
        """
        masked_emg, masked_actions => shape is (B, window_size, 8) and (B, window_size).
        The dataset does *not* do actual numeric masking on EMG,
        so `masked_emg` can be the raw or partially masked shape, and we do the real "embedding + mask" inside here.
        """
        batch_size = masked_emg.size(0)

        # ----------------- Main pipeline (dense) -----------------
        # linear projection
        masked_emg = self.emg_embedding(
            masked_emg.transpose(1, 2)
        )  # (B, embed_dim, W)
        if self.use_input_layernorm:
            masked_emg = self.emg_in_layer_norm(masked_emg.transpose(1, 2))
            masked_emg = masked_emg.transpose(1, 2)  # (B, embed_dim, W) again

        # Apply mask tokens if aligned
        if self.mask_alignment == "non-aligned":
            raise Exception(
                "non-aligned not implemented here for linear_projection"
            )
        elif self.mask_alignment == "aligned":
            # expand to (B, embed_dim, W)
            mask_tokens = self.linear_projection_learnable_mask.expand(
                masked_emg.shape[0], -1, masked_emg.shape[2]
            )
            mask_positions_emg_t = mask_positions_emg.transpose(1, 2)

            # Use channel 0's mask position is sufficient in the case where the mask positions are aligned.
            zeroth_embedding_values = mask_positions_emg_t[:, 0, :].unsqueeze(1)
            expanded_mask_positions = zeroth_embedding_values.expand(
                -1, self.embedding_dim, -1
            ).type_as(mask_tokens)
            masked_emg = (
                masked_emg * (1.0 - expanded_mask_positions)
                + mask_tokens * expanded_mask_positions
            )
        else:
            raise Exception(f"Unrecognized mask_alignment: {self.mask_alignment}")

        # Add modality-specific embedding & pos encoding
        masked_emg = masked_emg.transpose(1, 2)  # => (B, W, embed_dim)
        masked_emg = (
            masked_emg
            + self.emg_modality_specific_embedding["emg_id_embedding"]
            + self.emg_positional_encoding
        )

        # Action embedding
        action_embedded = (
            self.action_embedding(masked_actions)
            + self.action_modality_specific_embedding
            + self.action_positional_encoding
        )

        # Concatenate
        src = torch.cat([masked_emg, action_embedded], dim=1)
        seq_len = src.size(1)  # should be 2 * self.window_size

        # 6) ### ATTENTION MASK ###
        attention_mask = torch.zeros(
            (batch_size, seq_len, seq_len), dtype=torch.bool, device=src.device
        )
        # If task_idx == 3 => block all action positions
        # “action positions” here are the last window_size tokens
        for b in range(batch_size):
            if task_idx[b] == 3:
                # block [all queries from action, all keys from action]
                attention_mask[b, :, -self.window_size :] = True
                attention_mask[b, -self.window_size :, :] = True

        # Expand mask for multihead
        attention_mask = attention_mask.repeat_interleave(self.nhead, dim=0)

        # Transformer
        src = self.transformer_encoder(src, mask=attention_mask)

        if return_output:
            return src

        # Split back
        emg_encoded = src[:, : self.window_size, :]
        action_encoded = src[:, self.window_size :, :]

        # Project EMG
        emg_output = self.emg_output_projection(emg_encoded.transpose(1, 2))
        emg_output = emg_output.transpose(1, 2)  # => (B, W, 8)

        # Project Action
        if self.output_reduction_method == "none":
            action_output = self.action_output_projection(action_encoded)
        elif self.output_reduction_method == "pooling":
            pooled = self._reduce_sequence_by_pooling(
                action_encoded, self.chunk_size
            )
            action_output = self.action_output_projection(pooled)
        elif self.output_reduction_method == "learned":
            learned_agg = self._reduce_sequence_by_learned(
                action_encoded, self.chunk_size, self.chunk_aggregator
            )
            action_output = self.action_output_projection(learned_agg)
        else:
            raise ValueError(
                f"Unknown output_reduction_method: {self.output_reduction_method}"
            )

        return emg_output, action_output
