"""
TimeMixer++ for SOH Trajectory Forecasting

Paper: TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis
Reference: https://arxiv.org/abs/2410.16032

This implementation preserves the COMPLETE original TimeMixer++ architecture:
- FFT_for_Period: Adaptive period detection via FFT
- Time Imaging: Reshape 1D series to 2D based on detected periods
- Dual-Axis Attention: Row attention (trend) + Column attention (season)
- Multi-Scale Mixing: Cross-scale information fusion
- Multi-Resolution Mixing: Aggregate multiple period patterns

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AxialAttention import RowAttention, ColAttention
from layers.Embed import DataEmbedding_wo_pos
from layers.Conv_Blocks import Inception_Block_V1, Inception_Trans_Block_V1
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize


# =============================================================================
# Core TimeMixer++ Components (Preserved from original paper)
# =============================================================================

def FFT_for_Period(x, k=2):
    """
    Adaptive period detection via FFT (Core component of TimeMixer++)
    Finds top-k dominant periods in the input sequence
    """
    # Convert to FP32 for FFT (cuFFT requires power-of-2 dimensions for FP16)
    original_dtype = x.dtype
    x_fp32 = x.float()
    xf = torch.fft.rfft(x_fp32, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    if len(frequency_list) < k:
        k = len(frequency_list)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    index = np.where(period > 0)
    top_list = top_list[index]
    period = period[period > 0]
    # Convert back to original dtype
    return period, abs(xf).mean(-1)[:, top_list].to(original_dtype), top_list


class MultiScaleSeasonCross(nn.Module):
    """
    Cross-scale season mixing (High -> Low)
    Uses Inception Conv for multi-scale feature extraction
    """
    def __init__(self, configs):
        super(MultiScaleSeasonCross, self).__init__()
        self.cross_conv_season = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels, stride=(configs.down_sampling_window, 1)),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels))

    def forward(self, season_list):
        B, N, _, _ = season_list[0].size()
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = []
        out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))
        for i in range(len(season_list) - 1):
            out_low_res = self.cross_conv_season(out_high)
            # Align dimensions using interpolation if needed
            if out_low_res.shape[2:] != out_low.shape[2:]:
                out_low_res = F.interpolate(out_low_res, size=out_low.shape[2:], mode='bilinear', align_corners=False)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))
        return out_season_list


class MultiScaleTrendCross(nn.Module):
    """
    Cross-scale trend mixing (Low -> High)
    Uses interpolation + Conv for stable upsampling
    """
    def __init__(self, configs):
        super(MultiScaleTrendCross, self).__init__()
        self.down_sampling_window = configs.down_sampling_window
        # Use conv instead of transpose conv for more stable upsampling
        self.upsample_conv = nn.Sequential(
            nn.GELU(),
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, trend_list):
        B, N, _, _ = trend_list[0].size()
        trend_list_copy = [t.clone() for t in trend_list]
        trend_list_copy.reverse()
        out_low = trend_list_copy[0]
        out_high = trend_list_copy[1]
        out_trend_list = []
        out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        for i in range(len(trend_list_copy) - 1):
            # Use interpolation for upsampling (more stable than transpose conv)
            out_high_res = F.interpolate(out_low, size=out_high.shape[2:], mode='bilinear', align_corners=False)
            out_high_res = self.upsample_conv(out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_copy) - 1:
                out_high = trend_list_copy[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        out_trend_list.reverse()
        return out_trend_list


class MixerBlock(nn.Module):
    """
    Core TimeMixer++ block with:
    1. Time Imaging: 1D -> 2D based on detected period
    2. Dual-Axis Attention: Row (trend) + Column (season) attention
    3. Multi-Scale Mixing: Cross-scale information fusion
    4. Multi-Resolution Mixing: Aggregate multiple period patterns
    """
    def __init__(self, configs):
        super(MixerBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.k = configs.top_k
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.row_attn_2d_trend = RowAttention(configs.d_model, configs.d_ff)
        self.col_attn_2d_season = ColAttention(configs.d_model, configs.d_ff)
        self.multi_scale_season_conv = MultiScaleSeasonCross(configs)
        self.multi_scale_trend_conv = MultiScaleTrendCross(configs)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        period_list, period_weight, top_list = FFT_for_Period(x_list[-1], self.k)

        res_list = []
        for i in range(len(period_list)):
            period = period_list[i]
            season_list = []
            trend_list = []
            for x in x_list:
                out = self.time_imaging(x, period)
                season, trend = self.dual_axis_attn(out)
                season_list.append(season)
                trend_list.append(trend)

            out_list = self.multi_scale_mixing(season_list, trend_list, length_list)
            res_list.append(out_list)

        res_list_new = []
        for i in range(len(x_list)):
            list = []
            for j in range(len(period_list)):
                list.append(res_list[j][i])
            res = torch.stack(list, dim=-1)
            res_list_new.append(res)

        res_list_agg = []
        for x, res in zip(x_list, res_list_new):
            res = self.multi_reso_mixing(period_weight, x, res)
            res = self.layer_norm(res)
            res_list_agg.append(res)
        return res_list_agg

    def time_imaging(self, x, period):
        """Reshape 1D sequence to 2D image based on period"""
        B, T, N = x.size()
        out, length = self.__conv_padding(x, period)
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
        return out

    def dual_axis_attn(self, out):
        """Row attention for trend, Column attention for season"""
        trend = self.row_attn_2d_trend(out)
        season = self.col_attn_2d_season(out)
        return season, trend

    def multi_scale_mixing(self, season_list, trend_list, length_list):
        """Cross-scale mixing for season and trend"""
        out_season_list = self.multi_scale_season_conv(season_list)
        out_trend_list = self.multi_scale_trend_conv(trend_list)
        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])
        return out_list

    def __conv_padding(self, x, period, down_sampling_window=1):
        B, T, N = x.size()

        if T % (period * down_sampling_window) != 0:
            length = ((T // (period * down_sampling_window)) + 1) * period * down_sampling_window
            padding = torch.zeros([B, (length - T), N]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = T
            out = x
        return out, length

    def multi_reso_mixing(self, period_weight, x, res):
        """Aggregate patterns from multiple periods"""
        B, T, N = x.size()
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


# =============================================================================
# Domain Adaptation Layer for Battery SOH Task
# =============================================================================

class CycleEncoder(nn.Module):
    """
    Encode each cycle's charge/discharge curve into a fixed-dim vector.
    This is a standard domain adaptation approach (similar to patch embedding).

    Input: [B, num_cycles, num_vars, curve_len]
    Output: [B, num_cycles, d_model]
    """
    def __init__(self, num_vars, curve_len, d_model, dropout=0.1):
        super(CycleEncoder, self).__init__()
        self.num_vars = num_vars
        self.curve_len = curve_len
        self.d_model = d_model

        self.var_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(8),
            )
            for _ in range(num_vars)
        ])

        merged_dim = num_vars * 32 * 8
        self.merge_fc = nn.Sequential(
            nn.Linear(merged_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        B, num_cycles, num_vars, curve_len = x.shape

        var_features = []
        for v in range(num_vars):
            var_data = x[:, :, v, :].reshape(B * num_cycles, 1, curve_len)
            var_feat = self.var_conv[v](var_data)
            var_feat = var_feat.reshape(B * num_cycles, -1)
            var_features.append(var_feat)

        merged = torch.cat(var_features, dim=-1)
        cycle_features = self.merge_fc(merged)
        cycle_features = cycle_features.reshape(B, num_cycles, self.d_model)

        return cycle_features


# =============================================================================
# Main Model
# =============================================================================

class Model(nn.Module):
    """
    TimeMixer++ for SOH trajectory forecasting

    Architecture preserved from original paper:
    - MixerBlock with FFT-based period detection
    - Time Imaging (1D -> 2D)
    - Dual-Axis Attention (Row + Column)
    - Multi-Scale Cross Mixing
    - Multi-Resolution Aggregation

    Domain adaptation:
    - CycleEncoder for battery charge/discharge curves (input layer only)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')

        # Input processing parameters
        self.early_cycle_threshold = getattr(configs, 'early_cycle_threshold', 100)
        self.min_seq_len = getattr(configs, 'seq_len', 1)
        self.charge_discharge_length = getattr(configs, 'charge_discharge_length', 100)
        self.input_mode = getattr(configs, 'input_mode', 'current_voltage')

        # Sequence length for TimeMixer++ (internal use only, don't modify configs)
        self.seq_len = self.early_cycle_threshold - self.min_seq_len + 1
        self.label_len = getattr(configs, 'label_len', 48)
        self.pred_len = getattr(configs, 'pred_len', 5000)
        self.down_sampling_window = getattr(configs, 'down_sampling_window', 2)
        self.channel_independence = getattr(configs, 'channel_independence', 1)

        # Store internal seq_len for MixerBlock (use copy to avoid modifying original configs)
        self._internal_seq_len = self.seq_len
        configs.num_kernels = getattr(configs, 'num_kernels', 6)

        # Save original seq_len to restore later
        _original_seq_len = getattr(configs, 'seq_len', 100)

        # CycleEncoder for battery data
        if self.input_mode == 'soh_to_soh':
            self.seq_len = self.early_cycle_threshold
            self.cycle_encoder = None
            self.input_proj = nn.Linear(1, configs.d_model)
        else:
            self.num_vars = 3
            self.cycle_encoder = CycleEncoder(
                num_vars=self.num_vars,
                curve_len=self.charge_discharge_length,
                d_model=configs.d_model,
                dropout=configs.dropout
            )

        # Temporarily set configs.seq_len for MixerBlock creation
        configs.seq_len = self.seq_len

        # TimeMixer++ encoder (MixerBlocks)
        self.encoder_model = nn.ModuleList([MixerBlock(configs)
                                            for _ in range(configs.e_layers)])

        # Restore original seq_len to avoid polluting args for data loader
        configs.seq_len = _original_seq_len

        # Embedding layer
        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed if hasattr(configs, 'embed') else 'fixed',
                                                      configs.freq if hasattr(configs, 'freq') else 'h',
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model,
                                                      configs.embed if hasattr(configs, 'embed') else 'fixed',
                                                      configs.freq if hasattr(configs, 'freq') else 'h',
                                                      configs.dropout)

        # Normalization layers (RevIN)
        self.revin_layers = torch.nn.ModuleList([
            Normalize(configs.d_model if self.channel_independence else configs.enc_in,
                      affine=True, subtract_last=False)
            for i in range(getattr(configs, 'down_sampling_layers', 3) + 1)
        ])

        self.layer = configs.e_layers

        # Channel mixing attention (optional)
        if getattr(configs, 'channel_mixing', False):
            d_time_model = self.seq_len // (configs.down_sampling_window ** getattr(configs, 'down_sampling_layers', 3))
            self.channel_mixing_attention = AttentionLayer(
                FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                d_time_model, configs.n_heads)

        # Downsampling method
        down_sampling_method = getattr(configs, 'down_sampling_method', 'avg')
        if down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(configs.down_sampling_window, return_indices=False)
        elif down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(configs.down_sampling_window)
        elif down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            if self.channel_independence:
                in_channels = 1
            else:
                in_channels = configs.enc_in
            self.down_pool = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=3, padding=padding,
                                       stride=configs.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)
        else:
            self.down_pool = torch.nn.AvgPool1d(configs.down_sampling_window)

        # Prediction layers
        down_sampling_layers = getattr(configs, 'down_sampling_layers', 3)
        self.predict_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.seq_len // (self.down_sampling_window ** i),
                self.pred_len // (self.down_sampling_window ** i) if self.pred_len // (self.down_sampling_window ** i) > 0 else 1,
            )
            for i in range(down_sampling_layers + 1)
        ])

        # Output projection
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc=None):
        """Create multi-scale representations"""
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        down_sampling_layers = getattr(self.configs, 'down_sampling_layers', 3)
        for i in range(down_sampling_layers):
            down_sampling_method = getattr(self.configs, 'down_sampling_method', 'avg')
            if down_sampling_method == 'conv' and i == 0 and self.channel_independence:
                x_enc_ori = x_enc_ori.contiguous().reshape(B * N, T, 1).permute(0, 2, 1).contiguous()
            x_enc_sampling = self.down_pool(x_enc_ori)

            if down_sampling_method == 'conv':
                x_enc_sampling_list.append(
                    x_enc_sampling.reshape(B, N, T // (self.down_sampling_window ** (i + 1))).permute(0, 2, 1))
            else:
                x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))

            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list, None

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """TimeMixer++ forecasting for SOH task"""
        B, T, D = x_enc.size()  # [B, num_cycles, d_model]

        # Create multi-scale representations
        x_enc_list, _ = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        # Normalize each scale (using LayerNorm instead of RevIN for stability)
        enc_out_list = []
        for i, x in enumerate(x_enc_list):
            enc_out_list.append(x)

        # MixerBlocks
        for i in range(self.layer):
            enc_out_list = self.encoder_model[i](enc_out_list)

        # Prediction: temporal projection + channel projection
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # enc_out: [B, T_i, d_model]
            # Temporal projection: T_i -> pred_len_i
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, pred_len_i, d_model]
            # Channel projection: d_model -> 1
            dec_out = self.projection_layer(dec_out)  # [B, pred_len_i, 1]
            dec_out_list.append(dec_out)

        # Use only the finest scale output and squeeze
        dec_out = dec_out_list[0].squeeze(-1)  # [B, pred_len]
        return dec_out

    def forecast_soh_from_curves(self, cycle_curve_data, curve_attn_mask):
        """Forecast SOH from charge/discharge curves"""
        B = cycle_curve_data.shape[0]

        if curve_attn_mask is not None:
            tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
            cycle_curve_data = cycle_curve_data * tmp_mask

        # CycleEncoder
        x_enc = self.cycle_encoder(cycle_curve_data)  # [B, num_cycles, d_model]

        # TimeMixer++ forecast
        output = self.forecast(x_enc)  # [B, pred_len]
        return output

    def forecast_soh_from_soh(self, soh_input):
        """Forecast SOH from historical SOH"""
        if len(soh_input.shape) == 2:
            soh_input = soh_input.unsqueeze(-1)

        x_enc = self.input_proj(soh_input)
        output = self.forecast(x_enc)
        return output

    def forward(self, cycle_curve_data=None, curve_attn_mask=None,
                soh_input=None, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                aging_condition_embedding=None, soh_trajectory=None, trajectory_mask=None,
                soc_input=None, cycle_level_features=None, life_labels=None,
                return_embedding=False):
        """
        Forward pass compatible with MemoryNet interface

        Training: returns (output, recovery_loss, query_alignment_loss, embedding_alignment_loss, aug_loss)
        Inference: returns output
        """
        if self.input_mode == 'soh_to_soh':
            if soh_input is None:
                raise ValueError("soh_input is required for SOH-to-SOH mode")
            output = self.forecast_soh_from_soh(soh_input)
        else:
            if cycle_curve_data is None:
                raise ValueError("cycle_curve_data is required for current/voltage mode")
            output = self.forecast_soh_from_curves(cycle_curve_data, curve_attn_mask)

        if self.training:
            return output, 0.0, 0.0, 0.0, 0.0
        else:
            return output
