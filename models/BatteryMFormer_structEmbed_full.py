import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding
from layers.SelfAttention_Family import ACFullAttention, ACAttentionLayer
from layers.Transformer_EncDec import BattMDecoder, BattMDecoderLayer


# [STRUCT-EMBED MOD 1/3]

class StructuredConditionEmbed(nn.Module):
    def __init__(self, metadata_path: str, d_model: int, embed_dim: int = 32):
        super().__init__()
        meta = json.load(open(metadata_path, "r"))
        self.fields = meta["fields"]
        self.cardinalities = meta["cardinalities"]

        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(int(self.cardinalities[field]), embed_dim)
                for field in self.fields
            }
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * len(self.fields), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, 1, F] or [B, F]
        if ids.dim() == 3:
            ids = ids.squeeze(1)
        ids = ids.long()
        parts = []
        for idx, field in enumerate(self.fields):
            parts.append(self.embeddings[field](ids[:, idx]))
        x = torch.cat(parts, dim=-1)
        x = self.proj(x)
        return x.unsqueeze(1)


class SOCPatchEncoder(nn.Module):
    """Encodes intra-cycle charging/discharging curves using patch embeddings.

    The input curve for each cycle is segmented into patches along the time axis
    using a 1D convolution. For each patch index, features across cycles are
    aggregated via a small MLP to produce patch-level tokens.
    """

    def __init__(self, n_heads, d_model, d_ff, charge_discharge_length, early_cycle_threshold,
                 drop_rate, e_layers, kernel_size, cnn_channels, stride, enc_in):
        """Initializes the SOCPatchEncoder.

        Args:
            n_heads: Number of attention heads (kept for interface compatibility).
            d_model: Patch embedding dimension.
            d_ff: Hidden dimension for temporal aggregation MLP.
            charge_discharge_length: Fixed time-series length per cycle.
            early_cycle_threshold: Number of cycles used by the temporal aggregator
                (expects cycle dimension length to match this value).
            drop_rate: Dropout probability.
            e_layers: Number of layers (kept for interface compatibility).
            kernel_size: Patch length along the time axis.
            cnn_channels: CNN channel count (kept for interface compatibility).
            stride: Stride for patch extraction.
            enc_in: Number of input channels per curve.
        """
        super(SOCPatchEncoder, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.patch_len = kernel_size
        self.stride = stride
        self.fixed_len = charge_discharge_length
        self.enc_in = enc_in
        self.early_cycle_threshold = early_cycle_threshold

        self.num_patches = math.floor((self.fixed_len - self.patch_len) / self.stride) + 1

        self.patch_embedding = nn.Conv1d(
            in_channels=self.enc_in,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
            bias=False
        )
        self.patch_norm = nn.LayerNorm(self.d_model)

        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.early_cycle_threshold, self.d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.d_ff, self.d_ff),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.d_ff, 1)
        )

    def forward(self, x, curve_attn_mask):
        """Encodes cycle curves into patch tokens.

        Args:
            x: Cycle-curve tensor of shape [B, L, C, FL].
            curve_attn_mask: Valid-cycle mask of shape [B, L], where 1 indicates
                a valid cycle and 0 indicates padding.

        Returns:
            Patch tokens of shape [B, P, D].
        """
        B, L, C, FL = x.shape
        P = self.num_patches

        x = x.view(B * L, C, FL)                       # [B*L, C, FL]
        x_patch = self.patch_embedding(x)              # [B*L, D, P]

        x_inter = x_patch.view(B, L, self.d_model, P)  # [B, L, D, P]
        x_inter = x_inter.permute(0, 3, 1, 2).reshape(B * P, L, -1)  # [B*P, L, D]
        x_inter = self.patch_norm(x_inter)

        curve_attn_mask = curve_attn_mask.unsqueeze(1).expand(-1, P, -1)  # [B, P, L]
        curve_attn_mask = curve_attn_mask.reshape(B * P, L, 1)            # [B*P, L, 1]
        x_inter = x_inter * curve_attn_mask

        x_inter = x_inter.transpose(1, 2)              # [B*P, D, L]
        out = self.temporal_encoder(x_inter)           # [B*P, D, 1]

        out = out.reshape(B, P, self.d_model)          # [B, P, D]
        return out


class TrajectoryEncoder(nn.Module):
    """Encodes a future trajectory vector into a latent embedding."""

    def __init__(self, pred_len, d_ff, d_model, dropout):
        """Initializes the TrajectoryEncoder.

        Args:
            pred_len: Prediction horizon length.
            d_ff: Hidden dimension (kept for interface compatibility).
            d_model: Output embedding dimension.
            dropout: Dropout probability (kept for interface compatibility).
        """
        super(TrajectoryEncoder, self).__init__()
        self.pred_len = pred_len
        self.d_ff = d_ff
        self.d_model = d_model

        self.linear1 = nn.Linear(self.pred_len, self.pred_len // 2)
        self.linear2 = nn.Linear(self.pred_len // 2, self.pred_len // 4)
        self.linear3 = nn.Linear(self.pred_len // 4, self.pred_len // 8)
        self.linear4 = nn.Linear(self.pred_len // 8, self.d_model)

    def forward(self, x):
        """Encodes a trajectory vector.

        Args:
            x: Trajectory tensor of shape [B, pred_len].

        Returns:
            Latent embedding of shape [B, d_model].
        """
        h = self.linear1(x)
        h = F.gelu(h)

        h = self.linear2(h)
        h = F.gelu(h)

        h = self.linear3(h)
        h = F.gelu(h)

        out = self.linear4(h)
        return out


class MLPBlock(nn.Module):
    """Two-layer MLP with dropout and optional residual + LayerNorm."""

    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        """Initializes the MLP block.

        Args:
            in_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            out_dim: Output feature dimension.
            drop_rate: Dropout probability.
        """
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        """Applies the MLP block.

        Args:
            x: Input tensor of shape [..., in_dim].

        Returns:
            Output tensor of shape [..., out_dim].
        """
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)

        if x.shape[-1] == out.shape[-1]:
            out = self.ln(self.dropout(out) + x)
        else:
            out = self.ln(out)
        return out


class Model(nn.Module):
    """Memory-augmented model for SOH trajectory forecasting.

    The model encodes per-cycle signals and condition embeddings, decodes a set of
    learnable queries via cross-attention, retrieves relevant memory slots using
    sparse attention, and produces a horizon-length SOH prediction.
    """

    def __init__(self, configs, init_value_memory=None):
        """Initializes the model.

        Args:
            configs: Configuration object with required attributes.
            init_value_memory: Optional numpy array to initialize memory values.
        """
        super(Model, self).__init__()
        self.configs = configs

        self.seq_len = configs.seq_len
        self.top_k = configs.top_k if configs.top_k > 0 else None
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.d_ffs = configs.d_ffs
        self.d_llm = configs.d_llm
        self.n_heads = configs.n_heads if hasattr(configs, 'n_heads') else 4
        self.e_layers = configs.e_layers
        self.e_layers2 = configs.e_layers2
        self.d_layers = configs.d_layers if hasattr(configs, 'd_layers') else 1
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.activation = configs.activation if hasattr(configs, 'activation') else 'gelu'
        self.factor = configs.factor if hasattr(configs, 'factor') else 1
        self.charge_discharge_length = configs.charge_discharge_length if hasattr(configs, 'charge_discharge_length') else 100
        self.k_dim = configs.k_dim
        self.enc_in = configs.enc_in
        self.num_query = configs.num_query

        self.cnn_channels = configs.cnn_channels
        self.stride = configs.stride
        self.num_slots = configs.num_slots if hasattr(configs, 'num_slots') else 128
        self.temperature = configs.temperature if hasattr(configs, 'temperature') else 0.1
        self.num_segments = configs.num_segments

        self.norm_eps = 1e-6

        # [STRUCT-EMBED MOD 2/3]
        # Original:
        #   self.aging_condition_embed = nn.Linear(self.d_llm, self.d_model)
        metadata_path = getattr(configs, "structured_metadata_path", "")
        if not metadata_path:
            raise ValueError(
                "BatteryMFormer_structEmbed_full requires --structured_metadata_path"
            )
        embed_dim = int(getattr(configs, "structured_embed_dim", 32))
        self.aging_condition_embed = StructuredConditionEmbed(
            metadata_path, self.d_model, embed_dim=embed_dim
        )
        self.intra_flatten = nn.Flatten(start_dim=-2)

        self.soc_patch_encoder = SOCPatchEncoder(
            self.n_heads, self.d_model, self.d_ffs,
            self.charge_discharge_length, self.early_cycle_threshold,
            self.dropout, self.e_layers2, self.kernel_size,
            self.cnn_channels, self.stride, 4
        )

        self.intra_embed = nn.Sequential(
            nn.Linear(self.charge_discharge_length * 4, self.d_model)
        )

        self.intra_MLP = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, self.dropout)
            for _ in range(self.e_layers)
        ])

        self.pe = PositionalEmbedding(self.d_model)
        self.source_norm = nn.LayerNorm(self.d_model)

        self.feature_embed = nn.Linear(self.enc_in, self.d_model)
        self.learnable_queries = nn.Parameter(torch.rand(self.num_query, self.d_model))
        self.aging_condition_query_linear = nn.Linear(self.d_model, self.d_model)

        self.decoder = BattMDecoder(
            [
                BattMDecoderLayer(
                    ACAttentionLayer(
                        ACFullAttention(False, self.factor,
                                        attention_dropout=self.dropout,
                                        output_attention=False),
                        self.d_model,
                        self.n_heads
                    ),
                    ACAttentionLayer(
                        ACFullAttention(True, self.factor,
                                        attention_dropout=self.dropout,
                                        output_attention=False),
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.d_layers)
            ]
        )

        if init_value_memory is not None:
            self.M_val = nn.Parameter(torch.from_numpy(init_value_memory).float())
        else:
            self.M_val = nn.Parameter(torch.rand(self.num_slots, self.k_dim))

        self.trajectory_encoder = TrajectoryEncoder(self.pred_len, self.d_ff, self.k_dim, self.dropout)
        self.trajectory_decoder = nn.Linear(self.k_dim, self.pred_len)
        self.life_predictor = nn.Linear(self.k_dim, 1)

        self.gating_network = nn.Sequential(
            nn.Linear(self.k_dim * 2, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.k_dim),
            nn.Sigmoid()
        )
        self.gating_network[0].bias.data.fill_(0.0)
        self.gating_network[-2].bias.data.fill_(0.0)

        self.query_generator = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.Flatten(start_dim=-2),
            nn.Linear(self.d_ff * self.num_query, self.k_dim),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model * self.num_query, self.k_dim)
        )
        self.output_extrapolation = nn.Linear(self.k_dim, self.pred_len)

    def _unit_norm_keep_dir(self, x: torch.Tensor, eps: float = None) -> torch.Tensor:
        """Normalizes vectors to unit L2 norm without changing direction.

        The norm is computed in FP32 for mixed-precision stability and then cast
        back to the original dtype.

        Args:
            x: Input tensor with shape [..., D].
            eps: Small constant to avoid division by zero. If None, uses
                `self.norm_eps`.

        Returns:
            Tensor with the same shape and dtype as `x`, unit-normalized along the
            last dimension.
        """
        if eps is None:
            eps = self.norm_eps

        x_fp32 = x.float()
        n = x_fp32.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        y = x_fp32 / n
        return y.to(dtype=x.dtype)

    def compute_sparse_attention(self, query_keys, memory_keys, k=None, temperature=None):
        """Computes Top-K sparse attention weights between queries and memory slots.

        Similarity computation and softmax are performed in FP32 for numerical
        stability, and the result is cast back to the query dtype.

        Args:
            query_keys: Query tensor of shape [B, D].
            memory_keys: Memory tensor of shape [S, D].
            k: Number of memory slots to attend (Top-K). If None, uses all slots.
            temperature: Softmax temperature. If None, uses `self.temperature`.

        Returns:
            Sparse attention weights of shape [B, S]. Only Top-K entries per row
            are non-zero.
        """
        if k is None:
            k = self.num_slots
        if temperature is None:
            temperature = self.temperature

        out_dtype = query_keys.dtype

        S = int(memory_keys.size(0))
        k = int(min(max(int(k), 1), S))
        temperature = float(max(float(temperature), 1e-4))

        q = query_keys.float()
        m = memory_keys.float()

        qn = F.normalize(q, p=2, dim=-1, eps=1e-8)
        mn = F.normalize(m, p=2, dim=-1, eps=1e-8)

        sim = torch.matmul(qn, mn.t())
        topk_vals, topk_idx = torch.topk(sim, k, dim=-1)

        logits = topk_vals / temperature
        logits = logits - logits.max(dim=-1, keepdim=True).values

        topk_attn = F.softmax(logits, dim=-1)

        weights = torch.zeros_like(sim)
        weights.scatter_(-1, topk_idx, topk_attn.to(weights.dtype))

        return weights.to(out_dtype)

    def forward(self, cycle_curve_data, curve_attn_mask, soh_trajectory=None, trajectory_mask=None,
                return_components=False, aging_condition_embedding=None, soc_input=None, soh_input=None,
                cycle_level_features=None, life_labels=None, return_embedding=False):
        """Runs a forward pass.

        Args:
            cycle_curve_data: Cycle curve tensor of shape [B, L, C, FL].
            curve_attn_mask: Valid-cycle mask of shape [B, L], where 1 indicates valid.
            soh_trajectory: Ground-truth future SOH trajectory of shape [B, pred_len].
            trajectory_mask: Mask for trajectory values of shape [B, pred_len], where 1
                indicates valid.
            return_components: Reserved for future use (kept for interface compatibility).
            aging_condition_embedding: Condition embedding of shape [B, 1, d_llm].
            soc_input: SOC input to be appended as an extra channel, shape [B, L, FL].
            soh_input: Per-cycle SOH input features, shape [B, L, ...].
            cycle_level_features: Additional per-cycle features, shape [B, L, ...].
            life_labels: Lifetime regression targets, shape [B].
            return_embedding: If True, also return the embedding for domain adaptation.

        Returns:
            If training and return_embedding: tuple (output, recovery_loss, pred_life_loss,
                embedding_alignment_loss, 0.0, embedding).
            If training: tuple (output, recovery_loss, pred_life_loss,
                embedding_alignment_loss, 0.0).
            If not training and return_embedding: tuple (output, embedding).
            If not training: output tensor of shape [B, pred_len].
        """
        # [STRUCT-EMBED MOD 3/3]
        condition_embed = self.aging_condition_embed(aging_condition_embedding)  # [B, 1, d_model]
        B, L, C, fixed_len = cycle_curve_data.shape

        cycle_curve_data = torch.cat([cycle_curve_data, soc_input.unsqueeze(2)], dim=2)
        soc_informed_x = self.soc_patch_encoder(cycle_curve_data, curve_attn_mask)

        x = self.intra_flatten(cycle_curve_data)
        x = self.intra_embed(x)
        for mlp in self.intra_MLP:
            x = mlp(x)

        manual_features = torch.cat([soh_input, cycle_level_features], dim=-1)
        x = self.feature_embed(manual_features) + x

        source_x = torch.cat([x, soc_informed_x], dim=1)
        source_x = self.pe(source_x) + source_x
        source_x = self.source_norm(source_x)

        cross_mask = self._get_cross_attention_mask(
            torch.cat([curve_attn_mask, torch.ones_like(soc_informed_x[:, :, 0])], dim=-1),
            self.num_query
        )

        shared_queries = self.learnable_queries.unsqueeze(0).expand(B, -1, -1)
        decoder_input = shared_queries + self.aging_condition_query_linear(condition_embed)
        decoder_output = self.decoder(decoder_input, source_x, condition_embed=condition_embed, cross_mask=cross_mask)

        query = self.query_generator(decoder_output)
        query = self._unit_norm_keep_dir(query)

        M = self._unit_norm_keep_dir(self.M_val)
        alpha = self.compute_sparse_attention(query, M, k=self.top_k)
        V_mem = alpha @ M

        output = self.output_projection(decoder_output.reshape(B, -1))

        gate_input = torch.cat([output, V_mem], dim=-1)
        gate = self.gating_network(gate_input)
        output = output + gate * V_mem

        embedding = output  # [B, k_dim]
        output = self.output_extrapolation(output)

        if self.training:
            future_trajectory_embed = self.trajectory_encoder(soh_trajectory)
            future_trajectory_embed = self._unit_norm_keep_dir(future_trajectory_embed)

            v = F.normalize(V_mem.float(), dim=-1, eps=1e-8)
            t = F.normalize(future_trajectory_embed.float(), dim=-1, eps=1e-8)
            embedding_alignment_loss = (1 - (v * t).sum(dim=-1)).mean()

            future_trajectory_pred = self.trajectory_decoder(future_trajectory_embed)
            recovered_life_labels = self.life_predictor(future_trajectory_embed).squeeze(-1)
            pred_life_loss = F.mse_loss(recovered_life_labels, life_labels.float())

            recovery_loss = self.masked_mse_loss(future_trajectory_pred, soh_trajectory, trajectory_mask)
            if return_embedding:
                embedding = decoder_output.mean(dim=1)  # [B, d_model]
                return output, recovery_loss, pred_life_loss, embedding_alignment_loss, 0.0, embedding
            return output, recovery_loss, pred_life_loss, embedding_alignment_loss, 0.0

        if return_embedding:
            embedding = decoder_output.mean(dim=1)
            return output, embedding
        return output

    def masked_mse_loss(self, pred, target, mask):
        """Computes mean squared error over valid (unmasked) positions.

        Args:
            pred: Predicted values, shape [B, T].
            target: Target values, shape [B, T].
            mask: Validity mask, shape [B, T], where 1 indicates valid.

        Returns:
            Scalar loss value.
        """
        valid_pred = pred * mask
        valid_target = target * mask
        squared_error = (valid_pred - valid_target) ** 2
        denom = mask.sum().clamp_min(1.0)
        loss = squared_error.sum() / denom
        return loss

    def _get_cross_attention_mask(self, mask, query_length):
        """Builds a cross-attention mask for query-to-source attention.

        Args:
            mask: Source validity mask of shape [B, L], where 1 indicates valid.
            query_length: Number of query tokens.

        Returns:
            Boolean attention mask of shape [B, 1, query_length, L], where True
            indicates masked.
        """
        B, L = mask.shape
        attn_mask = mask.unsqueeze(1).unsqueeze(1)
        attn_mask = attn_mask.expand(-1, -1, query_length, -1)
        attn_mask = attn_mask == 0
        return attn_mask
