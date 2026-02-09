import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.individual = individual

        # Store key parameters
        self.charge_discharge_length = configs.charge_discharge_length if hasattr(configs, 'charge_discharge_length') else 300
        self.early_cycle_threshold = configs.early_cycle_threshold if hasattr(configs, 'early_cycle_threshold') else 100

        # Input dimensions based on mode
        if configs.input_mode == 'soh_to_soh':
            self.channels = 1  # Single variable (SOH)
            self.seq_len = self.early_cycle_threshold  # Use early_cycle_threshold for consistency
        else:
            self.channels = configs.enc_in if hasattr(configs, 'enc_in') else 3  # Voltage, Current, Capacity
            self.seq_len = self.early_cycle_threshold * self.charge_discharge_length

        # Output dimension (SOH trajectory length)
        self.pred_len = configs.pred_len if hasattr(configs, 'pred_len') else 5000

        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg if hasattr(configs, 'moving_avg') else 25)
        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                
                # Initialize weights
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
            # Initialize weights
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        
        # For multi-channel to single output (SOH)
        if self.channels > 1 and not self.individual:
            self.channel_projection = nn.Linear(self.channels, 1)
        else:
            self.channel_projection = None

    def encoder(self, x):
        """
        x: [Batch, Input length, Channel]
        """
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [Batch, Output length, Channel]

    def forecast(self, x_enc):
        """
        Forecast function for SOH trajectory
        x_enc: [B, seq_len, channels]
        """
        # Encoder with decomposition
        output = self.encoder(x_enc)  # [B, pred_len, channels]
        
        # Project to single channel for SOH if needed
        if self.channels > 1 and self.channel_projection is not None:
            output = self.channel_projection(output)  # [B, pred_len, 1]
        
        # Return as [B, pred_len] for SOH trajectory
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        
        return output

    def forward(self, cycle_curve_data=None, curve_attn_mask=None, soh_input=None,
                x_mark_enc=None, x_dec=None, x_mark_dec=None,
                aging_condition_embedding=None, soh_trajectory=None, trajectory_mask=None,
                soc_input=None, cycle_level_features=None, life_labels=None, return_embedding=False):
        """
        Forward pass supporting both input modes

        Training mode returns: (output, recovery_loss, query_alignment_loss, embedding_alignment_loss, aug_loss)
        Inference mode returns: output

        For current/voltage mode:
            cycle_curve_data: [B, early_cycles, num_vars, fixed_len]
            curve_attn_mask: [B, early_cycles]

        For SOH-to-SOH mode:
            soh_input: [B, seq_len, 1]
        """
        if self.configs.input_mode == 'soh_to_soh':
            # SOH-to-SOH prediction
            if soh_input is None:
                raise ValueError("soh_input is required for SOH-to-SOH mode")

            # Direct forecasting from SOH
            dec_out = self.forecast(soh_input)
        else:
            # Current/Voltage input mode
            if cycle_curve_data is None:
                raise ValueError("cycle_curve_data is required for current/voltage mode")

            # Apply masking to input (use multiplication instead of inplace operation)
            if curve_attn_mask is not None:
                tmp_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1)
                cycle_curve_data = cycle_curve_data * tmp_mask

            # Prepare input
            B, num_vars = cycle_curve_data.shape[0], cycle_curve_data.shape[2]
            cycle_curve_data = cycle_curve_data.transpose(2, 3)  # [B, L, fixed_len, num_vars]
            cycle_curve_data = cycle_curve_data.reshape(B, -1, num_vars)  # [B, L*fixed_len, num_vars]

            # Forecast
            dec_out = self.forecast(cycle_curve_data)

        # Training mode: return 5 values for compatibility with run_main.py
        if self.training:
            return dec_out, 0.0, 0.0, 0.0, 0.0
        else:
            return dec_out