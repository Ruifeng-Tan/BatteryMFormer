import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    (from original PatchMLP)
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block (from original PatchMLP)
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Encoder(nn.Module):
    """
    MLP Encoder block (from original PatchMLP)
    """

    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 2, 1)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)
        return y_2


class EmbLayer(nn.Module):
    """
    Multi-scale patch embedding layer (from original PatchMLP)
    """

    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model = max(1, d_model // patch_num)  # Ensure d_model >= 1 for long sequences
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model),
        )
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model * patch_num, d_model),
        )

    def forward(self, x):
        B, V, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)
        x = self.ff_1(x)
        return x


class Emb(nn.Module):
    """
    Multi-scale embedding module (from original PatchMLP)
    """

    def __init__(self, seq_len, d_model, patch_len=[48, 24, 12, 6]):
        super().__init__()
        patch_step = patch_len
        d_model_quarter = d_model // 4
        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0] // 2, seq_len, d_model_quarter)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1] // 2, seq_len, d_model_quarter)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2] // 2, seq_len, d_model_quarter)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3] // 2, seq_len, d_model_quarter)

    def forward(self, x):
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        s_out = torch.cat([s_x1, s_x2, s_x3, s_x4], -1)
        return s_out


class Model(nn.Module):
    """
    PatchMLP for SOH trajectory forecasting

    Architecture:
    - Input adapter: Flatten battery cycle curves
    - PatchMLP core: Instance norm + Multi-scale patch embedding + Series decomposition + MLP Encoder
    - Output adapter: Channel average + Linear projection to pred_len

    Input format:
    - current_voltage mode: cycle_curve_data [B, num_cycles, num_vars, curve_len]
    - soh_to_soh mode: soh_input [B, early_cycle_threshold, 1]
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name if hasattr(configs, 'task_name') else 'soh_forecast'
        self.input_mode = configs.input_mode if hasattr(configs, 'input_mode') else 'current_voltage'
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm if hasattr(configs, 'use_norm') else 1

        # Store config parameters
        self.min_seq_len = configs.seq_len
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.charge_discharge_length = configs.charge_discharge_length if hasattr(configs, 'charge_discharge_length') else 100

        # Calculate number of cycles in input
        self.num_cycles = self.early_cycle_threshold - self.min_seq_len + 1

        # Input dimensions based on mode
        if self.input_mode == 'soh_to_soh':
            self.enc_in = 1
            self.total_input_len = self.early_cycle_threshold
        else:
            self.enc_in = configs.enc_in if hasattr(configs, 'enc_in') else 3  # default 3: voltage, current, capacity
            self.total_input_len = self.num_cycles * self.charge_discharge_length

        # Output dimension (SOH trajectory length)
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 5000

        # Internal prediction length for PatchMLP (can be different from final pred_len)
        self.internal_pred_len = min(self.total_input_len // 2, 512)

        # PatchMLP Core Components
        # Series decomposition
        self.decomposition = series_decomp(13)

        # Multi-scale patch embedding - adjust patch_len based on input length
        if self.total_input_len >= 1000:
            patch_len = [48, 24, 12, 6]
        elif self.total_input_len >= 200:
            patch_len = [24, 12, 6, 3]
        else:
            patch_len = [12, 6, 4, 2]

        self.emb = Emb(self.total_input_len, configs.d_model, patch_len=patch_len)

        # MLP Encoder layers (seasonal and trend)
        e_layers = configs.e_layers if hasattr(configs, 'e_layers') else 3
        self.seasonal_layers = nn.ModuleList([
            Encoder(configs.d_model, self.enc_in)
            for _ in range(e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            Encoder(configs.d_model, self.enc_in)
            for _ in range(e_layers)
        ])

        # Internal projector (PatchMLP core output)
        self.projector = nn.Linear(configs.d_model, self.internal_pred_len, bias=True)

        # Output Adapter: project from internal_pred_len to pred_len
        self.output_projection = nn.Linear(self.internal_pred_len, self.pred_len)
        self.dropout = nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.1)

    def forecast(self, x_enc, attn_mask=None):
        """
        Core forecasting function using PatchMLP architecture

        Args:
            x_enc: [B, total_len, enc_in] - input time series
            attn_mask: [B, num_cycles] - optional mask (not used in MLP)

        Returns:
            dec_out: [B, pred_len] - predicted SOH trajectory
        """
        # Instance Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # [B, total_len, enc_in] -> [B, enc_in, total_len]
        x = x_enc.permute(0, 2, 1)

        # Multi-scale patch embedding: [B, enc_in, total_len] -> [B, enc_in, d_model]
        x = self.emb(x)

        # Series decomposition: separate trend and seasonal components
        seasonal_init, trend_init = self.decomposition(x)

        # MLP Encoder for seasonal component
        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)

        # MLP Encoder for trend component
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        # Combine seasonal and trend
        x = seasonal_init + trend_init

        # Project to internal prediction length: [B, enc_in, d_model] -> [B, enc_in, internal_pred_len]
        dec_out = self.projector(x)

        # [B, enc_in, internal_pred_len] -> [B, internal_pred_len, enc_in]
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalization
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.internal_pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.internal_pred_len, 1)

        # Average across channels: [B, internal_pred_len, enc_in] -> [B, internal_pred_len]
        dec_out = dec_out.mean(dim=-1)

        # Project to final pred_len: [B, internal_pred_len] -> [B, pred_len]
        dec_out = self.dropout(dec_out)
        dec_out = self.output_projection(dec_out)

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
            if soh_input is None:
                raise ValueError("soh_input is required for SOH-to-SOH mode")
            # soh_input: [B, early_cycle_threshold, 1]
            output = self.forecast(soh_input)
        else:
            if cycle_curve_data is None:
                raise ValueError("cycle_curve_data is required for current/voltage mode")

            # Mask unseen cycles (set to zeros)
            if curve_attn_mask is not None:
                tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
                cycle_curve_data = cycle_curve_data * tmp_mask

            # Reshape: [B, num_cycles, num_vars, curve_len] -> [B, total_len, num_vars]
            B = cycle_curve_data.shape[0]
            num_vars = cycle_curve_data.shape[2]
            # [B, num_cycles, num_vars, curve_len] -> [B, num_cycles, curve_len, num_vars]
            cycle_curve_data = cycle_curve_data.transpose(2, 3)
            # [B, num_cycles, curve_len, num_vars] -> [B, total_len, num_vars]
            cycle_curve_data = cycle_curve_data.reshape(B, -1, num_vars)

            output = self.forecast(cycle_curve_data, curve_attn_mask)

        # Training mode: return 5 values for compatibility with run_main.py
        if self.training:
            return output, 0.0, 0.0, 0.0, 0.0
        else:
            return output
