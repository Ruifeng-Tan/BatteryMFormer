import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
import math

class RoPEAttention(nn.Module):
    '''Self-Attention that uses Rotary PE'''
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(RoPEAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        queries, keys = self.RoPE(queries, keys)
        
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)
            else:
                scores.masked_fill_(attn_mask, -np.inf)
                

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device, dtype):
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(-1)
        # (output_dim//2)
        ids = torch.arange(0, output_dim // 2, dtype=dtype)  # i in [0, d/2]
        theta = torch.pow(10000, -2 * ids / output_dim)

        # (max_len, output_dim//2)
        embeddings = position * theta  # pos / (10000^(2i/d))

        # (max_len, output_dim//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (bs, head, max_len, output_dim//2, 2)
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

        # (bs, head, max_len, output_dim)
        # Interleaved: even=sin, odd=cos
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def RoPE(self, q, k):
        # q,k: (bs, head, max_len, output_dim)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        # (bs, head, max_len, output_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device, q.dtype)


        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        # Adjacent cos/sin pairs are identical, so repeat_interleave: (1,2,3) -> (1,1,2,2,3,3)
        cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # extract odd cols (cos) and duplicate
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # extract even cols (sin) and duplicate

        # q,k: (bs, head, max_len, output_dim)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)  # alternating sign pattern



        # Update q with RoPE
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)
        # Update k with RoPE
        k = k * cos_pos + k2 * sin_pos

        return q, k

        
class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)
            else:
                scores.masked_fill_(attn_mask, -np.inf)
                

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class ACFullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ACFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, condition_queries, attn_mask, tau=None, delta=None):
        # queries: [B, L, H, E]
        # keys:    [B, S, H, E]
        # values:  [B, S, H, D]
        # condition_queries: [B, 1, H, E] (Projected condition_embed)
        
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 1. Standard Attention Scores: Q * K^T
        # Shape: [B, H, L, S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 2. Aging Condition Bias (e): Condition_Q * K^T
        # Shape: [B, H, 1, S]
        # We use broadcast mechanism later: [B, H, L, S] + [B, H, 1, S]
        aging_scores = torch.einsum("bche,bshe->bhcs", condition_queries, keys)

        # 3. Apply Scaling to both (as requested: QK/sqrt(d) + e_scaled)
        # Note: Usually bias is added after scaling, but your request implies 
        # the condition score is also a result of a dot product operation 
        # that needs scaling to be comparable.
        scores = scale * scores
        aging_bias = scale * aging_scores

        # 4. Add the bias
        # This broadcasts the static battery condition attention across all time steps L
        scores = scores + aging_bias

        if self.mask_flag:
            if attn_mask is None:
                # Assuming TriangularCausalMask is defined elsewhere or imported
                # Using a simple triangular mask generation for standalone compatibility
                mask = torch.triu(torch.ones(L, S), diagonal=1).bool().to(queries.device)
                scores.masked_fill_(mask, -np.inf)
            else:
                scores.masked_fill_(attn_mask, -np.inf)

        # 5. Softmax and Weighting
        A = torch.softmax(scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ACAttentionLayer(nn.Module):
    '''
    Aging-aware attention
    '''
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(ACAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, condition_query, attn_mask, tau=None, delta=None):
        # queries: [B, L, D]
        # keys:    [B, S, D]
        # values:  [B, S, D]
        # condition_query: [B, D] (This is your condition_embed)

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 1. Project Standard Q, K, V
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 2. Project Condition Query (Shared Weights with W_Q)
        # Input: [B, D] -> Projected: [B, H*E] -> View: [B, 1, H, E]
        # We add a dimension '1' at dim=1 to represent the sequence length of the condition (which is 1)
        cond_queries = self.query_projection(condition_query).view(B, 1, H, -1)

        # 3. Pass everything to Inner Attention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            cond_queries, # Pass the projected condition
            attn_mask,
            tau=tau,
            delta=delta
        )
        
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
