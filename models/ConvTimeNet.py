"""
ConvTimeNet for SOH Trajectory Forecasting
Paper: https://arxiv.org/abs/2403.01493

Adapted from the original ConvTimeNet implementation.
Architecture kept unchanged: Deformable Patching + Depthwise Conv Encoder + Flatten Head.
Only input reshape and output channel aggregation are added for the battery SOH task.

Supports two input modes:
1. current_voltage: Input charge/discharge curves [B, num_cycles, num_vars, curve_len]
2. soh_to_soh: Input historical SOH [B, early_cycle_threshold, 1]
"""

import torch
from torch import nn
from layers.ConvTimeNet_backbone import ConvTimeNet_backbone


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.input_mode = getattr(configs, 'input_mode', 'current_voltage')
        self.pred_len = getattr(configs, 'pred_len', 5000)
        self.charge_discharge_length = getattr(configs, 'charge_discharge_length', 300)
        self.min_seq_len = configs.seq_len
        self.early_cycle_threshold = getattr(configs, 'early_cycle_threshold', 100)
        self.num_cycles = self.early_cycle_threshold - self.min_seq_len + 1

        # Input channels
        if self.input_mode == 'soh_to_soh':
            c_in = 1
            context_window = self.early_cycle_threshold
        else:
            c_in = 3  # Voltage, Current, Capacity (SOC is passed separately as soc_input)
            context_window = self.num_cycles * self.charge_discharge_length

        # ConvTimeNet hyperparameters
        patch_len = getattr(configs, 'patch_ks', 300)
        stride = getattr(configs, 'patch_sd', 300)
        n_layers = configs.e_layers
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        head_dropout = getattr(configs, 'head_dropout', 0.0)
        padding_patch = getattr(configs, 'padding_patch', 'end')

        # Depthwise conv kernel sizes (one per layer)
        dw_ks_str = getattr(configs, 'dw_ks', '11,15,21,29,39,51')
        if isinstance(dw_ks_str, str):
            dw_ks = [int(x) for x in dw_ks_str.split(',')]
        else:
            dw_ks = dw_ks_str
        # Ensure we have enough kernel sizes for all layers
        while len(dw_ks) < n_layers:
            dw_ks.append(dw_ks[-1])
        dw_ks = dw_ks[:n_layers]

        revin = bool(getattr(configs, 'revin', 0))
        affine = bool(getattr(configs, 'affine', 0))
        subtract_last = bool(getattr(configs, 'subtract_last', 0))
        deformable = bool(getattr(configs, 'deformable', 1))
        re_param = bool(getattr(configs, 're_param', 1))
        re_param_kernel = getattr(configs, 're_param_kernel', 3)
        enable_res_param = bool(getattr(configs, 'enable_res_param', 1))

        # Build backbone
        self.backbone = ConvTimeNet_backbone(
            c_in=c_in,
            seq_len=context_window,
            context_window=context_window,
            target_window=self.pred_len,
            patch_len=patch_len,
            stride=stride,
            n_layers=n_layers,
            dw_ks=dw_ks,
            d_model=d_model,
            d_ff=d_ff,
            norm='batch',
            dropout=dropout,
            act='gelu',
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            head_type='flatten',
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            deformable=deformable,
            enable_res_param=enable_res_param,
            re_param=re_param,
            re_param_kernel=re_param_kernel,
        )

        self.c_in = c_in

    def forecast_soh_from_curves(self, cycle_curve_data, curve_attn_mask):
        """
        Args:
            cycle_curve_data: [B, num_cycles, num_vars, curve_len]
            curve_attn_mask: [B, num_cycles] - 1 for visible, 0 for masked
        Returns:
            output: [B, pred_len]
        """
        B = cycle_curve_data.shape[0]

        # Mask padding cycles
        tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
        cycle_curve_data = cycle_curve_data * tmp_mask

        # Reshape: [B, num_cycles, num_vars, curve_len] -> [B, num_vars, total_len]
        cycle_curve_data = cycle_curve_data.permute(0, 2, 1, 3)  # [B, num_vars, num_cycles, curve_len]
        z = cycle_curve_data.reshape(B, self.c_in, -1)  # [B, num_vars, total_len]

        # ConvTimeNet backbone -> [B, num_vars, pred_len]
        z = self.backbone(z)

        # Channel aggregation: mean across variables
        output = z.mean(dim=1)  # [B, pred_len]
        return output

    def forecast_soh_from_soh(self, soh_input):
        """
        Args:
            soh_input: [B, early_cycle_threshold, 1]
        Returns:
            output: [B, pred_len]
        """
        if len(soh_input.shape) == 2:
            soh_input = soh_input.unsqueeze(-1)

        # [B, L, 1] -> [B, 1, L]
        z = soh_input.permute(0, 2, 1)

        # ConvTimeNet backbone -> [B, 1, pred_len]
        z = self.backbone(z)

        # Remove channel dim
        output = z.squeeze(1)  # [B, pred_len]
        return output

    def forward(self, cycle_curve_data=None, curve_attn_mask=None, soh_input=None,
                x_mark_enc=None, x_dec=None, x_mark_dec=None,
                aging_condition_embedding=None, soh_trajectory=None, trajectory_mask=None,
                soc_input=None, cycle_level_features=None, life_labels=None,
                return_embedding=False):
        """
        Unified forward interface for MemoryNet framework.

        Training returns: (output, 0.0, 0.0, 0.0, 0.0)
        Inference returns: output [B, pred_len]
        """
        if self.input_mode == 'soh_to_soh':
            output = self.forecast_soh_from_soh(soh_input)
        else:
            output = self.forecast_soh_from_curves(cycle_curve_data, curve_attn_mask)

        if self.training:
            return output, 0.0, 0.0, 0.0, 0.0
        else:
            return output
