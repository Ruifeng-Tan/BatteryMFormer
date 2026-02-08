"""
Loss functions for Physical Prior parameter prediction

Supports two loss types:
1. Parameter-space loss: Compare predicted vs ground truth parameters (RMSE, R-squared)
2. SOH-space loss: Convert parameters to SOH and compare trajectories (MAPE, RMSE, MAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ParameterLoss(nn.Module):
    """
    Parameter-space loss: Compare predicted vs ground truth parameters directly

    Loss = RMSE(params_pred, params_gt)

    This loss focuses on learning the correct parameter values
    """

    def __init__(self):
        super(ParameterLoss, self).__init__()

    def forward(self, params_pred, params_gt, mask=None):
        """
        Args:
            params_pred: [B, 4] - predicted parameters (normalized)
            params_gt: [B, 4] - ground truth parameters (normalized)
            mask: [B] - optional binary mask (1 for valid, 0 for ignore)

        Returns:
            loss: scalar
            metrics: dict with RMSE and R-squared per parameter
        """
        B = params_pred.shape[0]

        if mask is not None:
            # Apply mask
            params_pred = params_pred * mask.unsqueeze(1)
            params_gt = params_gt * mask.unsqueeze(1)
            valid_count = mask.sum().clamp(min=1)
        else:
            valid_count = B

        # Compute RMSE for each parameter
        mse_per_param = ((params_pred - params_gt) ** 2).mean(dim=0)  # [4]
        rmse_per_param = torch.sqrt(mse_per_param)

        # Overall RMSE (average over all parameters)
        rmse_total = rmse_per_param.mean()

        # Compute R-squared for each parameter
        r2_per_param = []
        for i in range(4):
            pred_i = params_pred[:, i]
            gt_i = params_gt[:, i]

            ss_res = ((gt_i - pred_i) ** 2).sum()
            ss_tot = ((gt_i - gt_i.mean()) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_per_param.append(r2)

        r2_per_param = torch.stack(r2_per_param)

        # Metrics
        metrics = {
            'rmse_total': rmse_total.item(),
            'rmse_A': rmse_per_param[0].item(),
            'rmse_B': rmse_per_param[1].item(),
            'rmse_C': rmse_per_param[2].item(),
            'rmse_D': rmse_per_param[3].item(),
            'r2_A': r2_per_param[0].item(),
            'r2_B': r2_per_param[1].item(),
            'r2_C': r2_per_param[2].item(),
            'r2_D': r2_per_param[3].item()
        }

        return rmse_total, metrics


class SOHLoss(nn.Module):
    """
    SOH-space loss: Convert parameters to SOH trajectory and compare with ground truth

    This requires computing SOH from parameters for cycles 101 to EOL
    Loss is computed only in the prediction region [101, EOL]
    """

    def __init__(self, max_cycle=5200, truncate_start_cycle=100, eol_threshold=0.8):
        super(SOHLoss, self).__init__()
        self.max_cycle = max_cycle
        self.truncate_start_cycle = truncate_start_cycle
        self.eol_threshold = eol_threshold

    def params_to_soh(self, params, max_cycle):
        """
        Convert parameters to SOH trajectory

        Args:
            params: [B, 4] - parameters (A, B, C, D) in SCALED space
            max_cycle: int - maximum cycle number

        Returns:
            soh_trajectory: [B, max_cycle] - SOH values for cycles 1 to max_cycle
        """
        B = params.shape[0]
        device = params.device

        # Create cycle numbers [1, 2, 3, ..., max_cycle]
        cycles = torch.arange(1, max_cycle + 1, device=device).float()  # [max_cycle]

        # Normalize cycles to [0, 1]
        N_norm = cycles / max_cycle  # [max_cycle]

        # Expand to batch
        N_norm = N_norm.unsqueeze(0).repeat(B, 1)  # [B, max_cycle]

        # Extract parameters
        A = params[:, 0].unsqueeze(1)  # [B, 1]
        B_param = params[:, 1].unsqueeze(1)
        C = params[:, 2].unsqueeze(1)
        D = params[:, 3].unsqueeze(1)

        # Compute SOH: SOH(N_norm) = A + B*N_norm + C*N_norm^2 + D*N_norm^3
        N_norm_2 = N_norm ** 2
        N_norm_3 = N_norm ** 3

        soh = A + B_param * N_norm + C * N_norm_2 + D * N_norm_3

        return soh

    def forward(self, params_pred, soh_gt, trajectory_mask):
        """
        Args:
            params_pred: [B, 4] - predicted parameters (in SCALED normalized space)
            soh_gt: [B, max_trajectory_len] - ground truth SOH trajectory (normalized to [0,1])
            trajectory_mask: [B, max_trajectory_len] - mask (1 for 101-EOL, 0 elsewhere)

        Returns:
            loss: scalar
            metrics: dict with MAPE, RMSE, MAE
        """
        B = params_pred.shape[0]
        max_trajectory_len = soh_gt.shape[1]

        # Convert parameters to SOH trajectory
        soh_pred = self.params_to_soh(params_pred, max_trajectory_len)  # [B, max_trajectory_len]

        # Apply mask (only compute loss in prediction region 101-EOL)
        mask = trajectory_mask  # [B, max_trajectory_len]

        # Masked MSE loss
        mse_loss = ((soh_pred - soh_gt) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        rmse_loss = torch.sqrt(mse_loss)

        # Compute metrics (MAPE, RMSE, MAE) on masked region
        # NOTE: soh_pred and soh_gt are already in real SOH scale (0.8 to 1.5+)
        # No normalization/denormalization needed

        # Masked metrics
        with torch.no_grad():
            # MAPE
            abs_percentage_error = torch.abs((soh_gt - soh_pred) / (soh_gt + 1e-8)) * mask
            mape = (abs_percentage_error.sum() / mask.sum().clamp(min=1) * 100).item()

            # RMSE
            squared_error = ((soh_gt - soh_pred) ** 2 * mask).sum()
            rmse = (torch.sqrt(squared_error / mask.sum().clamp(min=1))).item()

            # MAE
            abs_error = (torch.abs(soh_gt - soh_pred) * mask).sum()
            mae = (abs_error / mask.sum().clamp(min=1)).item()

        metrics = {
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        }

        return rmse_loss, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss: Weighted combination of parameter loss and SOH loss

    Loss = alpha * param_loss + beta * soh_loss
    """

    def __init__(self, alpha=0.5, beta=0.5, max_cycle=5200, truncate_start_cycle=100, eol_threshold=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.param_loss_fn = ParameterLoss()
        self.soh_loss_fn = SOHLoss(max_cycle, truncate_start_cycle, eol_threshold)

    def forward(self, params_pred, params_gt, soh_gt, trajectory_mask, param_mask=None):
        """
        Args:
            params_pred: [B, 4]
            params_gt: [B, 4]
            soh_gt: [B, max_trajectory_len]
            trajectory_mask: [B, max_trajectory_len]
            param_mask: [B] - optional

        Returns:
            total_loss: scalar
            param_metrics: dict
            soh_metrics: dict
        """
        # Parameter loss
        param_loss, param_metrics = self.param_loss_fn(params_pred, params_gt, param_mask)

        # SOH loss
        soh_loss, soh_metrics = self.soh_loss_fn(params_pred, soh_gt, trajectory_mask)

        # Combined loss
        total_loss = self.alpha * param_loss + self.beta * soh_loss

        return total_loss, param_metrics, soh_metrics
