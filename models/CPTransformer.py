"""
CPTransformer for SOH Trajectory Forecasting
Supports two input modes:
1. SOH to SOH: Input historical SOH, predict future SOH trajectory
2. Current/Voltage to SOH: Input charge/discharge curves, predict SOH trajectory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from layers.Autoformer_EncDec import series_decomp

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
    CPTransformer for SOH trajectory forecasting
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  # 100
        self.pred_len = configs.pred_len  # 5200
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers if hasattr(configs, 'd_layers') else 1
        self.dropout = configs.dropout
        self.activation = configs.activation if hasattr(configs, 'activation') else 'gelu'
        self.factor = configs.factor if hasattr(configs, 'factor') else 1
        
        # For handling different input modes
        self.charge_discharge_length = configs.charge_discharge_length if hasattr(configs, 'charge_discharge_length') else 100
        self.early_cycle_threshold = configs.early_cycle_threshold if hasattr(configs, 'early_cycle_threshold') else 100
        
        # === Intra-cycle processing (for current/voltage input) ===
        self.intra_flatten = nn.Flatten(start_dim=2)
        self.intra_embed = nn.Linear(self.charge_discharge_length * 3, self.d_model)
        self.intra_MLP = nn.ModuleList([
            MLPBlock(self.d_model, self.d_ff, self.d_model, self.dropout) 
            for _ in range(self.e_layers)
        ])
        
        # === SOH input embedding (for SOH to SOH mode) ===
        self.soh_embed = nn.Linear(1, self.d_model)
        
        # === Series decomposition (optional, from Autoformer) ===
        self.decomposition = series_decomp(25)
        
        # === Positional encoding ===
        self.pe = PositionalEmbedding(self.d_model)
        
        # === Transformer Encoder for temporal modeling ===
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, self.factor, 
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
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        

        # === Prediction heads ===
        # For current_voltage mode: num_visible_cycles = early_cycle_threshold - seq_len + 1
        self.num_visible_cycles = self.early_cycle_threshold - self.seq_len + 1
        self.output_projection = nn.Linear(self.d_model * self.num_visible_cycles, self.pred_len)

        # For soh_to_soh mode: input length is early_cycle_threshold
        self.soh_output_projection = nn.Linear(self.d_model * self.early_cycle_threshold, self.pred_len)
        
    def forecast_soh_from_curves(self, cycle_curve_data, curve_attn_mask):
        """
        Forecast SOH trajectory from charge/discharge curves
        Args:
            cycle_curve_data: [B, early_cycles, num_vars, fixed_len]
            curve_attn_mask: [B, early_cycles]
        Returns:
            SOH predictions: [B, pred_len]
        """
        B = cycle_curve_data.shape[0]
        # Mask padding
        tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
        cycle_curve_data = cycle_curve_data * tmp_mask
        
        # Process each cycle with intra-cycle MLPs
        cycle_curve_data = self.intra_flatten(cycle_curve_data)  # [B, early_cycles, features]
        cycle_embeddings = self.intra_embed(cycle_curve_data)  # [B, early_cycles, d_model]
        
        for i in range(self.e_layers):
            cycle_embeddings = self.intra_MLP[i](cycle_embeddings)
        
        # Add positional encoding
        cycle_embeddings = self.pe(cycle_embeddings) + cycle_embeddings
        
        # Prepare attention mask for encoder
        enc_self_mask = self._get_attention_mask(curve_attn_mask)
        
        # Encode with Transformer
        enc_out, _ = self.encoder(cycle_embeddings, attn_mask=enc_self_mask)
        
        # Project along temporal dimension
        enc_out = enc_out.reshape(B, -1)  # [B, num_visible_cycles * d_model]
        output = self.output_projection(enc_out)  # [B, pred_len]
        return output
    
    def forecast_soh_from_soh(self, soh_input):
        """
        Forecast SOH trajectory from historical SOH values
        Args:
            soh_input: [B, early_cycle_threshold, 1] or [B, early_cycle_threshold]
        Returns:
            SOH predictions: [B, pred_len]
        """
        if len(soh_input.shape) == 2:
            soh_input = soh_input.unsqueeze(-1)  # [B, early_cycle_threshold, 1]

        B = soh_input.shape[0]

        # Embed SOH values
        soh_embeddings = self.soh_embed(soh_input)  # [B, early_cycle_threshold, d_model]

        # Add positional encoding
        soh_embeddings = self.pe(soh_embeddings) + soh_embeddings

        # Encode with Transformer
        enc_out, _ = self.encoder(soh_embeddings)

        # Flatten and project to pred_len
        enc_out = enc_out.reshape(B, -1)  # [B, early_cycle_threshold * d_model]
        output = self.soh_output_projection(enc_out)  # [B, pred_len]

        return output
    
    def _get_attention_mask(self, mask):
        """
        Convert padding mask to attention mask
        Args:
            mask: [B, L] with 1 for valid, 0 for padding
        Returns:
            attention_mask: [B, 1, L, L] with True for positions to mask
        """
        B, L = mask.shape
        # Create causal mask
        attn_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
        attn_mask = attn_mask.repeat(1, 1, L, 1)  # [B, 1, L, L]
        # Convert to boolean (True means mask)
        attn_mask = (attn_mask == 0)
        return attn_mask
    
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