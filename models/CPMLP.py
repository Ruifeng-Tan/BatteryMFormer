"""
CPMLP for SOH Trajectory Forecasting
Supports two input modes:
1. SOH to SOH: Input historical SOH, predict future SOH trajectory
2. Current/Voltage to SOH: Input charge/discharge curves, predict SOH trajectory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate):
        super(MLPBlock, self).__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.out_linear = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        '''
        x: [B, *, in_dim]
        '''
        out = self.in_linear(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.out_linear(out)
        # Residual connection if dimensions match
        if x.shape[-1] == out.shape[-1]:
            out = self.ln(self.dropout(out) + x)
        else:
            out = self.ln(out)
        return out


class Model(nn.Module):
    """
    CPMLP for SOH trajectory forecasting
    Pure MLP-based architecture without attention mechanisms
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  # minimum input cycles
        self.pred_len = configs.pred_len  # prediction length (e.g., 5000)
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.e_layers = configs.e_layers  # Number of intra-cycle MLP layers
        self.d_layers = configs.d_layers if hasattr(configs, 'd_layers') else 1  # Number of inter-cycle MLP layers

        # For handling different input modes
        self.charge_discharge_length = configs.charge_discharge_length if hasattr(configs, 'charge_discharge_length') else 100
        self.early_cycle_threshold = configs.early_cycle_threshold if hasattr(configs, 'early_cycle_threshold') else 100

        # Number of visible cycles (consistent with CPTransformer)
        self.num_visible_cycles = self.early_cycle_threshold - self.seq_len + 1

        # === Intra-cycle processing (for current/voltage input) ===
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(self.charge_discharge_length * 3, self.d_model)
        self.intra_MLP = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, self.dropout)
            for _ in range(self.e_layers)
        ])

        # === SOH input processing (for SOH to SOH mode) ===
        self.soh_embed = nn.Linear(1, self.d_model)

        # === Inter-cycle processing ===
        # Consistent with CPTransformer: use (early_cycle_threshold - seq_len + 1) cycles
        self.inter_flatten = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_visible_cycles * self.d_model, self.d_model)
        )

        self.inter_MLP = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, self.dropout)
            for _ in range(self.d_layers)
        ])

        # === Output projection (consistent with CPTransformer) ===
        self.output_projection = nn.Linear(self.d_model, self.pred_len)
        
    def forecast_soh_from_curves(self, cycle_curve_data, curve_attn_mask):
        """
        Forecast SOH trajectory from charge/discharge curves
        Args:
            cycle_curve_data: [B, early_cycles, num_vars, fixed_len]
            curve_attn_mask: [B, early_cycles]
        Returns:
            SOH predictions: [B, pred_len]
        """
        # Mask padding cycles
        tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
        cycle_curve_data = cycle_curve_data * tmp_mask

        # Process each cycle with intra-cycle MLPs
        cycle_curve_data = self.intra_flatten(cycle_curve_data)  # [B, early_cycles, features]
        cycle_embeddings = self.intra_embed(cycle_curve_data)  # [B, early_cycles, d_model]

        # Apply intra-cycle MLP blocks
        for i in range(self.e_layers):
            cycle_embeddings = self.intra_MLP[i](cycle_embeddings)

        # Flatten and process inter-cycle relationships
        inter_features = self.inter_flatten(cycle_embeddings)  # [B, d_model]

        # Apply inter-cycle MLP blocks
        for i in range(self.d_layers):
            inter_features = self.inter_MLP[i](inter_features)

        # Output projection
        output = self.output_projection(F.relu(inter_features))  # [B, pred_len]
        return output

    def forecast_soh_from_soh(self, soh_input):
        """
        Forecast SOH trajectory from historical SOH values
        Args:
            soh_input: [B, seq_len, 1] or [B, seq_len]
        Returns:
            SOH predictions: [B, pred_len]
        """
        if len(soh_input.shape) == 2:
            soh_input = soh_input.unsqueeze(-1)  # [B, seq_len, 1]

        # Embed SOH values
        soh_embeddings = self.soh_embed(soh_input)  # [B, seq_len, d_model]

        # Apply intra-cycle MLP blocks
        for i in range(self.e_layers):
            soh_embeddings = self.intra_MLP[i](soh_embeddings)

        # Flatten and process temporal relationships
        inter_features = self.inter_flatten(soh_embeddings)  # [B, d_model]

        # Apply inter-cycle MLP blocks
        for i in range(self.d_layers):
            inter_features = self.inter_MLP[i](inter_features)

        # Output projection
        output = self.output_projection(F.relu(inter_features))  # [B, pred_len]
        return output

    def forward(self, cycle_curve_data=None, curve_attn_mask=None,
                soh_input=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                aging_condition_embedding=None, soh_trajectory=None, trajectory_mask=None,
                soc_input=None, cycle_level_features=None, life_labels=None,
                return_embedding=False):
        """
        Forward pass supporting both input modes

        Training mode returns: (output, recovery_loss, query_alignment_loss, embedding_alignment_loss, aug_loss)
        Inference mode returns: output
        """
        if self.configs.input_mode != 'current_voltage':
            # SOH to SOH mode
            output = self.forecast_soh_from_soh(soh_input)
        else:
            # Current/Voltage to SOH mode
            output = self.forecast_soh_from_curves(cycle_curve_data, curve_attn_mask)

        # Training mode: return 5 values for compatibility with run_main.py
        if self.training:
            return output, 0.0, 0.0, 0.0, 0.0
        else:
            return output