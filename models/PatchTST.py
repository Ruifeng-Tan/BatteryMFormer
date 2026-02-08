import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class Model(nn.Module):
    """
    PatchTST for SOH trajectory forecasting
    Modified to support both current/voltage input and SOH-to-SOH prediction

    Input format:
    - current_voltage mode: cycle_curve_data [B, num_cycles, num_vars, curve_len]
      where num_cycles = early_cycle_threshold - seq_len + 1
    - soh_to_soh mode: soh_input [B, early_cycle_threshold, 1]
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.input_mode = configs.input_mode if hasattr(configs, 'input_mode') else 'current_voltage'
        self.d_model = configs.d_model

        # Store config parameters
        self.min_seq_len = configs.seq_len  # Minimum input cycles
        self.early_cycle_threshold = configs.early_cycle_threshold  # Maximum input cycles
        self.charge_discharge_length = configs.charge_discharge_length if hasattr(configs, 'charge_discharge_length') else 100

        # Calculate number of cycles in input
        self.num_cycles = self.early_cycle_threshold - self.min_seq_len + 1

        # Input dimensions based on mode
        if self.input_mode == 'soh_to_soh':
            self.enc_in = 1  # Single variable (SOH)
            self.total_input_len = self.early_cycle_threshold  # SOH sequence length
        else:
            self.enc_in = 4  # Voltage, Current, Capacity, SOC (4 channels in dataloader)
            self.total_input_len = self.num_cycles * self.charge_discharge_length

        # Output dimension (SOH trajectory length)
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 5200

        # Patch parameters
        self.patch_len = configs.patch_len if hasattr(configs, 'patch_len') else 16
        self.stride = configs.stride if hasattr(configs, 'stride') else 8
        padding = self.stride

        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, padding, configs.dropout)
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # Prediction Head for SOH trajectory
        # Calculate patch_num based on actual input length
        self.patch_num = int((self.total_input_len - self.patch_len) / self.stride + 2)
        self.head_nf = configs.d_model * self.patch_num

        # Direct projection from flattened features to pred_len
        self.output_projection = nn.Linear(self.head_nf, self.pred_len)
        self.dropout_layer = nn.Dropout(configs.dropout)

    def forecast_soh_from_curves(self, x_enc, curve_attn_mask=None):
        """
        Forecast SOH trajectory from charge/discharge curves

        Args:
            x_enc: [B, total_len, enc_in] - flattened cycle curves
            curve_attn_mask: [B, num_cycles] - 1 for visible, 0 for masked

        Returns:
            dec_out: [B, pred_len] - predicted SOH trajectory
        """
        B = x_enc.shape[0]

        # Normalization (Instance Norm)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, enc_in, total_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*enc_in, patch_num, d_model]

        # Apply attention mask if provided
        attn_mask = None
        if curve_attn_mask is not None:
            # Expand cycle mask to point-level
            point_mask = torch.repeat_interleave(
                curve_attn_mask, dim=1, repeats=self.charge_discharge_length
            )  # [B, total_len]
            point_mask = torch.repeat_interleave(point_mask, dim=0, repeats=n_vars)  # [B*n_vars, total_len]

            # Convert to patch mask
            point_mask = self.padding_patch_layer(point_mask)
            patch_mask = point_mask.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            patch_mask = torch.sum(patch_mask, dim=-1)  # [B*n_vars, patch_num]
            patch_mask = (patch_mask >= 1).float()

            # Create 2D attention mask
            patch_mask = patch_mask.unsqueeze(1).unsqueeze(1)  # [B*n_vars, 1, 1, patch_num]
            patch_mask = patch_mask.expand(-1, -1, patch_mask.shape[-1], -1)
            attn_mask = (patch_mask == 0)  # True = masked

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=attn_mask)

        # Reshape: [B*n_vars, patch_num, d_model] -> [B, n_vars, d_model, patch_num]
        enc_out = torch.reshape(enc_out, (B, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, n_vars, d_model, patch_num]

        # Average across variables
        enc_out = enc_out.mean(dim=1)  # [B, d_model, patch_num]

        # Flatten and project to pred_len
        enc_out = enc_out.reshape(B, -1)  # [B, d_model * patch_num]
        enc_out = self.dropout_layer(enc_out)
        dec_out = self.output_projection(enc_out)  # [B, pred_len]

        return dec_out

    def forecast_soh_from_soh(self, x_enc):
        """
        Forecast SOH trajectory from historical SOH

        Args:
            x_enc: [B, early_cycle_threshold, 1] - input SOH sequence

        Returns:
            dec_out: [B, pred_len] - predicted SOH trajectory
        """
        B = x_enc.shape[0]

        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, 1, early_cycle_threshold]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Reshape
        enc_out = torch.reshape(enc_out, (B, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, n_vars=1, d_model, patch_num]

        # Remove n_vars dimension and flatten
        enc_out = enc_out.squeeze(1)  # [B, d_model, patch_num]
        enc_out = enc_out.reshape(B, -1)  # [B, d_model * patch_num]
        enc_out = self.dropout_layer(enc_out)
        dec_out = self.output_projection(enc_out)  # [B, pred_len]

        return dec_out

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
            curve_attn_mask: [B, num_cycles] - 1 for visible cycles, 0 for masked

        For SOH-to-SOH mode:
            soh_input: [B, early_cycle_threshold, 1]
        """
        if self.input_mode == 'soh_to_soh':
            # SOH-to-SOH prediction
            if soh_input is None:
                raise ValueError("soh_input is required for SOH-to-SOH mode")
            output = self.forecast_soh_from_soh(soh_input)
        else:
            # Current/Voltage input mode
            if cycle_curve_data is None:
                raise ValueError("cycle_curve_data is required for current/voltage mode")

            # Mask unseen cycles (set to zeros)
            tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
            cycle_curve_data = cycle_curve_data * tmp_mask

            # Reshape: [B, num_cycles, num_vars, curve_len] -> [B, total_len, num_vars]
            B = cycle_curve_data.shape[0]
            num_vars = cycle_curve_data.shape[2]
            cycle_curve_data = cycle_curve_data.transpose(2, 3)  # [B, num_cycles, curve_len, num_vars]
            cycle_curve_data = cycle_curve_data.reshape(B, -1, num_vars)  # [B, num_cycles*curve_len, num_vars]

            output = self.forecast_soh_from_curves(cycle_curve_data, curve_attn_mask)

        # Training mode: return 5 values for compatibility with run_main.py
        if self.training:
            return output, 0.0, 0.0, 0.0, 0.0
        else:
            return output
