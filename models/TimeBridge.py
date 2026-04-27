"""
TimeBridge: Non-Stationarity Meets Multi-Scale Time Series Forecasting

Adapted from the original TimeBridge implementation for battery SOH trajectory
prediction. All vendor layers are inlined into this single file.

Reference:
    https://github.com/hqh0728/TimeBridge
"""
from __future__ import annotations

import copy
import math
from math import sqrt
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Battery adapter utilities
# ---------------------------------------------------------------------------

def _ensure_soh_dim(soh_input: torch.Tensor) -> torch.Tensor:
    if soh_input.dim() == 2:
        soh_input = soh_input.unsqueeze(-1)
    return soh_input


def _build_battery_ms_sequence(
    cycle_curve_data: torch.Tensor | None,
    curve_attn_mask: torch.Tensor | None,
    soh_input: torch.Tensor | None,
    input_mode: str,
) -> torch.Tensor:
    """Convert battery batch tensors to a generic [B, T, C] multivariate sequence."""
    if input_mode == "soh_to_soh":
        if soh_input is None:
            raise ValueError("soh_input is required for soh_to_soh mode")
        return _ensure_soh_dim(soh_input)

    if cycle_curve_data is None or soh_input is None:
        raise ValueError("cycle_curve_data and soh_input are required for current_voltage mode")

    if curve_attn_mask is not None:
        curve_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1).to(cycle_curve_data.dtype)
        cycle_curve_data = cycle_curve_data * curve_mask
        soh_mask = curve_attn_mask.unsqueeze(-1).to(soh_input.dtype)
    else:
        soh_mask = None

    bsz, num_cycles, num_vars, curve_len = cycle_curve_data.shape
    curve_features = cycle_curve_data.reshape(bsz, num_cycles, num_vars * curve_len)

    soh_seq = _ensure_soh_dim(soh_input)
    if soh_mask is not None:
        soh_seq = soh_seq * soh_mask

    return torch.cat([curve_features, soh_seq], dim=-1)


def _build_zero_time_marks(x_enc: torch.Tensor) -> torch.Tensor:
    return torch.zeros((x_enc.shape[0], x_enc.shape[1], 4), dtype=x_enc.dtype, device=x_enc.device)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, args, num_p=1, d_model=None):
        super().__init__()
        self.num_p = num_p
        self.patch = args.seq_len // self.num_p
        self.d_model = args.d_model if d_model is None else d_model

        self.proj = nn.Sequential(
            nn.Linear(self.patch, self.d_model, False),
            nn.Dropout(args.dropout)
        )

    def forward(self, x, x_mark):
        x = torch.cat([x, x_mark], dim=-1).transpose(-1, -2)
        x = self.proj(x.reshape(*x.shape[:-1], self.num_p, self.patch))
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None, **kwargs):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A


class TSMixer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        q = self.q_proj(q).reshape(B, L, H, -1)
        k = self.k_proj(k).reshape(B, S, H, -1)
        v = self.v_proj(v).reshape(B, S, H, -1)

        out, attn = self.attention(q, k, v, res=res, attn=attn)
        out = out.view(B, L, -1)
        return self.out(out), attn


# ---------------------------------------------------------------------------
# Encoder layers
# ---------------------------------------------------------------------------

def PeriodNorm(x, period_len=6):
    if len(x.shape) == 3:
        x = x.unsqueeze(-2)
    b, c, n, t = x.shape
    x_patch = [x[..., period_len - 1 - i:-i + t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    mean = F.pad(mean.reshape(b * c, n, -1),
                 mode='replicate', pad=(period_len - 1, 0)).reshape(b, c, n, -1)
    out = x - mean
    return out.squeeze(-2)


class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns


class IntAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, stable_len=8,
                 dropout=0.1, activation="relu", stable=True, enc_in=None, **kwargs):
        super().__init__()
        self.stable = stable
        self.stable_len = stable_len
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self._temporal_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))
        return self.norm2(x + y), None

    def _temporal_attn(self, x):
        b, c, n, d = x.shape
        new_x = x.reshape(-1, n, d)
        qk = new_x
        if self.stable:
            with torch.no_grad():
                qk = PeriodNorm(new_x, self.stable_len)
        new_x = self.attention(qk, qk, new_x)[0]
        return new_x.reshape(b, c, n, d)


class PatchSampling(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 in_p=30, out_p=4, stable=False, stable_len=8):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.in_p = in_p
        self.out_p = out_p
        self.stable = stable
        self.stable_len = stable_len

        self.attention = attention
        self.conv1 = nn.Conv1d(self.in_p, self.out_p, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv1d(self.out_p + 1, self.out_p, 1, 1, 0, bias=False)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self._down_attn(x)
        y = x = self.norm1(new_x)

        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))
        return self.norm2(x + y), None

    def _down_attn(self, x):
        b, c, n, d = x.shape
        x = x.reshape(-1, n, d)
        new_x = self.conv1(x)
        new_x = self.conv2(torch.cat(
            [new_x, x.mean(-2, keepdim=True)], dim=-2)) + new_x
        new_x = self.attention(new_x, x, x)[0] + self.dropout(new_x)
        return new_x.reshape(b, c, -1, d)


class CointAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, axial=True, stable_len=8,
                 dropout=0.1, activation="relu", stable=True, enc_in=None, **kwargs):
        super().__init__()
        self.stable = stable
        self.stable_len = stable_len
        d_ff = d_ff or 4 * d_model

        self.axial_func = axial
        self.attention1 = attention
        self.attention2 = copy.deepcopy(attention)

        self.num_rc = math.ceil((enc_in + 4) ** 0.5)
        self.pad_ch = nn.ConstantPad1d((0, self.num_rc ** 2 - (enc_in + 4)), 0)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.axial_func:
            new_x = self._axial_attn(x)
        else:
            new_x = self._full_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))
        return self.norm2(x + y), None

    def _axial_attn(self, x):
        b, c, n, d = x.shape
        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = (self.pad_ch(new_x.transpose(-1, -2))
                 .transpose(-1, -2).reshape(-1, self.num_rc, d))
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b r) c d -> (b c) r d', r=self.num_rc)
        new_x = self.attention2(new_x, new_x, new_x)[0] + new_x
        new_x = rearrange(new_x, '(b n c) r d -> b (r c) n d', b=b, n=n)
        return new_x[:, :c, ...]

    def _full_attn(self, x):
        b, c, n, d = x.shape
        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b n) c d -> b c n d', b=b, n=n)
        return new_x[:, :c, :]


# ---------------------------------------------------------------------------
# TimeBridge backbone
# ---------------------------------------------------------------------------

class TimeBridgeBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        self.embedding = PatchEmbed(configs, num_p=self.num_p)

        layers = self._layers_init(configs)
        self.encoder = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )

    def _layers_init(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for _ in range(configs.ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for _ in range(configs.ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        x_enc = self.embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(x_enc)[0][:, :self.c_in, ...]
        dec_out = self.decoder(enc_out).transpose(-1, -2)
        return dec_out * std + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


# ---------------------------------------------------------------------------
# Battery wrapper (public interface)
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    TimeBridge adapted for battery SOH trajectory prediction.

    Converts the battery batch format to a generic [B, T, C] multivariate
    forecasting interface consumed by the TimeBridge backbone.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.input_mode = getattr(configs, "input_mode", "current_voltage")
        self.seq_len = getattr(configs, "early_cycle_threshold", 100)
        self.pred_len = getattr(configs, "pred_len", 5000)

        if self.input_mode == "current_voltage":
            self.enc_in = getattr(configs, "enc_in", 3) * getattr(configs, "charge_discharge_length", 300) + 1
        else:
            self.enc_in = 1

        tb_cfg = SimpleNamespace(
            revin=getattr(configs, "use_revin", 1),
            enc_in=self.enc_in,
            period=10,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            num_p=None,
            d_model=getattr(configs, "d_model", 128),
            d_ff=max(getattr(configs, "d_ff", 256), getattr(configs, "d_model", 128)),
            n_heads=getattr(configs, "n_heads", 4),
            attn_dropout=getattr(configs, "dropout", 0.1),
            dropout=getattr(configs, "dropout", 0.1),
            stable_len=6,
            ia_layers=max(1, min(getattr(configs, "e_layers", 2), 2)),
            pd_layers=1,
            ca_layers=1,
            activation=getattr(configs, "activation", "gelu"),
        )
        self.backbone = TimeBridgeBackbone(tb_cfg)

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
        cycle_level_features=None,
        life_labels=None,
        return_embedding=False,
    ):
        x_enc = _build_battery_ms_sequence(
            cycle_curve_data=cycle_curve_data,
            curve_attn_mask=curve_attn_mask,
            soh_input=soh_input,
            input_mode=self.input_mode,
        )
        if x_mark_enc is None:
            x_mark_enc = _build_zero_time_marks(x_enc)
        dec_out = self.backbone(x_enc, x_mark_enc, None, None)
        output = dec_out[:, :, -1]
        if self.training:
            return output, 0.0, 0.0, 0.0, 0.0
        return output
