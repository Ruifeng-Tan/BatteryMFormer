import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name

        # Model dimensions
        self.d_model = configs.d_model
        self.d_llm = configs.d_llm
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 5000

        # Store key parameters
        self.charge_discharge_length = configs.charge_discharge_length if hasattr(configs, 'charge_discharge_length') else 300
        self.early_cycle_threshold = configs.early_cycle_threshold if hasattr(configs, 'early_cycle_threshold') else 100

        # Input dimensions based on mode.
        # For current/voltage forecasting, the raw BatteryLife curve has enc_in channels
        # (typically voltage/current/capacity). This rebuttal variant appends SOC as an
        # extra channel during forward(), and also prepends one condition token before the
        # encoder. We keep those counts explicit to avoid projection-size mismatches.
        if configs.input_mode == 'soh_to_soh':
            self.enc_in = 1
            self.curve_enc_in = 1
            self.output_tokens = 1
            self.seq_len = self.early_cycle_threshold
        else:
            self.enc_in = configs.enc_in if hasattr(configs, 'enc_in') else 3
            self.curve_enc_in = self.enc_in + 1  # + SOC channel
            self.output_tokens = self.curve_enc_in + 1  # + condition token
            self.min_seq_len = configs.seq_len
            self.num_cycles = self.early_cycle_threshold - self.min_seq_len + 1
            self.seq_len = self.num_cycles * self.charge_discharge_length

        # Inverted Embedding: sequence length -> d_model for each variable
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,
            configs.d_model,
            configs.embed if hasattr(configs, 'embed') else 'timeF',
            configs.freq if hasattr(configs, 'freq') else 'h',
            configs.dropout
        )

        # Aging condition embedding MLP
        self.aging_condition_embed_mlp = nn.Sequential(
            nn.Linear(self.d_llm, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor if hasattr(configs, 'factor') else 1,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation if hasattr(configs, 'activation') else 'gelu'
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Output projections for the two input modes.
        self.act = F.gelu
        self.dropout_layer = nn.Dropout(configs.dropout)
        self.output_projection = nn.Linear(configs.d_model * self.output_tokens, self.pred_len)
        self.soh_output_projection = nn.Linear(configs.d_model, self.pred_len)

    def forecast_soh_from_curves(self, x_enc, condition_embed):
        """
        Forecast SOH trajectory from charge/discharge curves

        Args:
            x_enc: [B, num_cycles * curve_len, enc_in] - flattened cycle curves
            condition_embed: [B, 1, d_model] - aging condition embedding

        Returns:
            output: [B, pred_len] - predicted SOH trajectory
        """
        # Instance Normalization for numerical stability in fp16
        # Normalizes each variable independently across the sequence dimension
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Inverted embedding: [B, seq_len, enc_in] -> [B, enc_in, d_model]
        enc_out = self.enc_embedding(x_enc, None)

        # Concatenate condition embedding with encoder output
        # enc_out: [B, enc_in, d_model], condition_embed: [B, 1, d_model]
        # We add condition embedding as an additional "expert" dimension
        B = enc_out.shape[0]
        # Expand condition_embed to match enc_in dimension: [B, enc_in, d_model]
        enc_out = torch.cat([condition_embed, enc_out], dim=1)

        # Encoder: [B, enc_in, d_model] -> [B, enc_in, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Activation and dropout (following BatteryLife)
        output = self.act(enc_out)
        output = self.dropout_layer(output)

        # Flatten: [B, output_tokens, d_model] -> [B, output_tokens * d_model]
        output = output.reshape(output.shape[0], -1)

        # Project to pred_len: [B, enc_in * d_model] -> [B, pred_len]
        output = self.output_projection(output)

        return output

    def forecast_soh_from_soh(self, x_enc):
        """
        Forecast SOH trajectory from historical SOH values

        Args:
            x_enc: [B, early_cycle_threshold, 1] - input SOH sequence

        Returns:
            output: [B, pred_len] - predicted SOH trajectory
        """
        # Instance Normalization for numerical stability in fp16
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Inverted embedding: [B, seq_len, 1] -> [B, 1, d_model]
        enc_out = self.enc_embedding(x_enc, None)

        # Encoder
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Activation and dropout
        output = self.act(enc_out)
        output = self.dropout_layer(output)

        # Flatten: [B, 1, d_model] -> [B, d_model]
        output = output.reshape(output.shape[0], -1)

        # Project to pred_len: [B, d_model] -> [B, pred_len]
        output = self.soh_output_projection(output)

        return output

    def forward(self, cycle_curve_data=None, curve_attn_mask=None, soh_input=None,
                x_mark_enc=None, x_dec=None, x_mark_dec=None,
                aging_condition_embedding=None, soh_trajectory=None, trajectory_mask=None,
                soc_input=None, cycle_level_features=None, life_labels=None,
                return_embedding=False):
        """
        Forward pass supporting both input modes

        Training mode returns: (output, recovery_loss, query_alignment_loss, embedding_alignment_loss, aug_loss)
        Inference mode returns: output

        For current/voltage mode:
            cycle_curve_data: [B, num_cycles, num_vars, curve_len]
            curve_attn_mask: [B, num_cycles] - 1 for visible, 0 for masked

        For SOH-to-SOH mode:
            soh_input: [B, early_cycle_threshold, 1]
        """
        if self.configs.input_mode == 'soh_to_soh':
            # SOH-to-SOH prediction
            if soh_input is None:
                raise ValueError("soh_input is required for SOH-to-SOH mode")
            output = self.forecast_soh_from_soh(soh_input)
        else:
            # Current/Voltage input mode
            if cycle_curve_data is None:
                raise ValueError("cycle_curve_data is required for current/voltage mode")

            # Concatenate SOC input to cycle_curve_data
            cycle_curve_data = torch.cat([cycle_curve_data, soc_input.unsqueeze(2)], dim=2)

            # Apply masking: set unseen cycles to zeros
            if curve_attn_mask is not None:
                tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
                cycle_curve_data = cycle_curve_data * tmp_mask

            # Get condition embedding from aging_condition_embedding
            condition_embed = self.aging_condition_embed_mlp(aging_condition_embedding)
            if condition_embed.dim() == 2:
                condition_embed = condition_embed.unsqueeze(1)

            # Reshape: [B, num_cycles, num_vars, curve_len] -> [B, num_cycles * curve_len, num_vars]
            B = cycle_curve_data.shape[0]
            num_vars = cycle_curve_data.shape[2]
            cycle_curve_data = cycle_curve_data.transpose(2, 3)  # [B, num_cycles, curve_len, num_vars]
            x_enc = cycle_curve_data.reshape(B, -1, num_vars)  # [B, num_cycles * curve_len, num_vars]

            output = self.forecast_soh_from_curves(x_enc, condition_embed)

        # Training mode: return 5 values for compatibility with run_main.py
        if self.training:
            return output, 0.0, 0.0, 0.0, 0.0
        else:
            return output
