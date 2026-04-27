import json
from pathlib import Path

import torch
import torch.nn as nn

from models.BatteryMFormer import Model as BaseBatteryMFormer


class StructuredConditionEmbed(nn.Module):
    def __init__(self, metadata_path: str, d_model: int, embed_dim: int = 32):
        super().__init__()
        meta = json.load(open(metadata_path, "r"))
        self.fields = meta["fields"]
        self.cardinalities = meta["cardinalities"]

        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(int(self.cardinalities[field]), embed_dim)
                for field in self.fields
            }
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * len(self.fields), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, 1, F] or [B, F]
        if ids.dim() == 3:
            ids = ids.squeeze(1)
        ids = ids.long()
        parts = []
        for idx, field in enumerate(self.fields):
            parts.append(self.embeddings[field](ids[:, idx]))
        x = torch.cat(parts, dim=-1)
        x = self.proj(x)
        return x.unsqueeze(1)


class Model(BaseBatteryMFormer):
    def __init__(self, configs, init_value_memory=None):
        super().__init__(configs, init_value_memory=init_value_memory)
        metadata_path = getattr(configs, "structured_metadata_path", "")
        if not metadata_path:
            raise ValueError("BatteryMFormer_structEmbed requires --structured_metadata_path")
        embed_dim = int(getattr(configs, "structured_embed_dim", 32))
        self.aging_condition_embed = StructuredConditionEmbed(metadata_path, self.d_model, embed_dim=embed_dim)
