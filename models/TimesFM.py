"""
TimesFM Baseline for SOH Trajectory Prediction (Zero-shot)

This module wraps Google's TimesFM foundation model for battery SOH prediction.
No fine-tuning required - uses zero-shot inference.

Usage:
    model = TimesFMBaseline(context_length=100, pred_len=500)
    output = model(soh_input, curve_attn_mask, ...)
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add TSorchestra to path
sys.path.insert(0, '/ai/dl_project/TSorchestra')

from src.models.foundation.timesfm import TimesFM


class Model(nn.Module):
    """
    TimesFM Baseline wrapper for MemoryNet SOH prediction.

    This model uses Google's TimesFM foundation model for zero-shot
    time series forecasting. It converts MemoryNet's input format to
    TimesFM's expected format and returns predictions.

    Args:
        configs: Configuration object with following attributes:
            - pred_len: Maximum prediction length
            - seq_len: Minimum input sequence length
            - early_cycle_threshold: Maximum input sequence length
            - context_length: TimesFM context length (default: 2048)
            - batch_size: Inference batch size (default: 64)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = getattr(configs, 'pred_len', 500)
        self.seq_len = getattr(configs, 'seq_len', 100)
        self.early_cycle_threshold = getattr(configs, 'early_cycle_threshold', 100)

        # TimesFM parameters
        self.context_length = getattr(configs, 'context_length', 2048)
        self.tfm_batch_size = getattr(configs, 'tfm_batch_size', 64)
        self.repo_id = getattr(configs, 'timesfm_repo_id', 'google/timesfm-2.5-200m-pytorch')

        # Initialize TimesFM model (lazy loading)
        self._timesfm = None

    @property
    def timesfm(self):
        """Lazy load TimesFM model to avoid loading at import time."""
        if self._timesfm is None:
            self._timesfm = TimesFM(
                repo_id=self.repo_id,
                context_length=self.context_length,
                batch_size=self.tfm_batch_size,
                alias='TimesFM'
            )
        return self._timesfm

    def _prepare_dataframe(
        self,
        soh_input: torch.Tensor,
        curve_attn_mask: torch.Tensor = None
    ) -> pd.DataFrame:
        """
        Convert MemoryNet input format to TimesFM DataFrame format.

        Args:
            soh_input: [B, seq_len, 1] or [B, seq_len] SOH input values
            curve_attn_mask: [B, seq_len] mask (1=visible, 0=masked)

        Returns:
            DataFrame with columns: unique_id, ds, y
        """
        # Handle tensor conversion
        if isinstance(soh_input, torch.Tensor):
            soh_input = soh_input.detach().cpu().numpy()

        if curve_attn_mask is not None and isinstance(curve_attn_mask, torch.Tensor):
            curve_attn_mask = curve_attn_mask.detach().cpu().numpy()

        # Squeeze if needed
        if soh_input.ndim == 3:
            soh_input = soh_input.squeeze(-1)  # [B, seq_len]

        batch_size, seq_len = soh_input.shape

        rows = []
        for i in range(batch_size):
            # Determine valid length from mask
            if curve_attn_mask is not None:
                valid_len = int(curve_attn_mask[i].sum())
            else:
                valid_len = seq_len

            # Create rows for this battery
            for j in range(valid_len):
                rows.append({
                    'unique_id': f'battery_{i}',
                    'ds': pd.Timestamp('2020-01-01') + pd.Timedelta(days=j),
                    'y': float(soh_input[i, j])
                })

        return pd.DataFrame(rows)

    def _parse_forecast(
        self,
        forecast_df: pd.DataFrame,
        batch_size: int,
        pred_len: int
    ) -> np.ndarray:
        """
        Parse TimesFM forecast output back to tensor format.

        Args:
            forecast_df: TimesFM output DataFrame
            batch_size: Number of batteries
            pred_len: Prediction length

        Returns:
            numpy array of shape [B, pred_len]
        """
        output = np.zeros((batch_size, pred_len))

        for i in range(batch_size):
            battery_id = f'battery_{i}'
            battery_forecast = forecast_df[forecast_df['unique_id'] == battery_id]

            if len(battery_forecast) > 0:
                # Get TimesFM predictions
                pred_values = battery_forecast['TimesFM'].values
                # Truncate or pad to pred_len
                actual_len = min(len(pred_values), pred_len)
                output[i, :actual_len] = pred_values[:actual_len]
                # Pad with last value if needed
                if actual_len < pred_len:
                    output[i, actual_len:] = pred_values[-1] if len(pred_values) > 0 else 0.8

        return output

    def forward(
        self,
        cycle_curve_data=None,
        curve_attn_mask=None,
        soh_input=None,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        aging_condition_embedding=None,
        soh_trajectory=None,
        trajectory_mask=None,
        soc_input=None,
        return_embedding=False
    ):
        """
        Forward pass using TimesFM for SOH prediction.

        For soh_to_soh mode:
            - soh_input: [B, seq_len, 1] input SOH values
            - curve_attn_mask: [B, seq_len] visibility mask

        Returns:
            Training: (output, 0.0, 0.0, 0.0, 0.0) - compatible with MemoryNet loss interface
            Inference: output [B, pred_len]
        """
        # Determine input source
        if soh_input is not None:
            # soh_to_soh mode
            input_data = soh_input
            mask = curve_attn_mask
        else:
            raise ValueError("TimesFM baseline only supports soh_to_soh mode. Please provide soh_input.")

        # Get batch size
        if isinstance(input_data, torch.Tensor):
            batch_size = input_data.shape[0]
            device = input_data.device
        else:
            batch_size = input_data.shape[0]
            device = torch.device('cpu')

        # Prepare DataFrame for TimesFM
        df = self._prepare_dataframe(input_data, mask)

        # Run TimesFM forecast
        forecast_df = self.timesfm.forecast(
            df=df,
            h=self.pred_len,
            freq='D'  # Use daily frequency (cycles as days)
        )

        # Parse output
        output_np = self._parse_forecast(forecast_df, batch_size, self.pred_len)

        # Convert to tensor
        output = torch.from_numpy(output_np).float().to(device)

        # Return format compatible with MemoryNet training loop
        if self.training:
            # Return (output, recovery_loss, query_alignment_loss, embedding_alignment_loss, aug_loss)
            return output, 0.0, 0.0, 0.0, 0.0
        else:
            return output


class TimesFMEnsemble(nn.Module):
    """
    TSorchestra Ensemble Baseline for SOH prediction.

    Uses multiple foundation models (TimesFM, Moirai, etc.) with
    SLSQP weight optimization for ensemble forecasting.
    """

    def __init__(self, configs):
        super(TimesFMEnsemble, self).__init__()
        self.configs = configs
        self.pred_len = getattr(configs, 'pred_len', 500)
        self.context_length = getattr(configs, 'context_length', 2048)

        # Lazy loading
        self._ensemble = None

    @property
    def ensemble(self):
        """Lazy load ensemble model."""
        if self._ensemble is None:
            from src.models.foundation.timesfm import TimesFM
            from src.models.foundation.moirai import Moirai
            from src.models.ensembles.slsqp import SLSQPEnsemble

            models = [
                TimesFM(repo_id='google/timesfm-2.5-200m-pytorch'),
                Moirai(repo_id='Salesforce/moirai-1.1-R-base'),
            ]
            self._ensemble = SLSQPEnsemble(models=models, metric='mae')
        return self._ensemble

    def forward(self, soh_input, curve_attn_mask=None, **kwargs):
        """Forward pass using ensemble."""
        # Similar implementation to TimesFMBaseline
        # ... (omitted for brevity)
        raise NotImplementedError("Ensemble mode not yet implemented. Use TimesFMBaseline instead.")
