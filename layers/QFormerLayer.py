import torch
import torch.nn as nn

class QFormerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        
        # 1. Self Attention (Queries talking to Queries)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # 2. Cross Attention (Queries talking to Image Features)
        # This is the unique part inserted into the BERT architecture
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # 3. Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_embeds, image_embeds, query_mask=None):
        """
        query_embeds: [Batch, Num_Queries, Hidden_Size] (e.g., 32 learned queries)
        image_embeds: [Batch, Num_Patches, Hidden_Size] (Output from frozen ViT)
        """
        
        # --- Block 1: Self Attention ---
        residual = query_embeds 
        
        # Queries attend to themselves (with optional causal/bidirectional masking)
        attn_output, _ = self.self_attn(query=query_embeds, key=query_embeds, value=query_embeds, attn_mask=query_mask)
        
        # ✅ RESIDUAL CONNECTION 1
        query_embeds = residual + self.dropout(attn_output)
        query_embeds = self.norm1(query_embeds)

        # --- Block 2: Cross Attention ---
        residual = query_embeds
        
        # Queries (Q) attend to Image Features (K, V)
        attn_output, _ = self.cross_attn(query=query_embeds, key=image_embeds, value=image_embeds)
        
        # ✅ RESIDUAL CONNECTION 2
        query_embeds = residual + self.dropout(attn_output)
        query_embeds = self.norm2(query_embeds)

        # --- Block 3: Feed Forward ---
        residual = query_embeds
        
        ffn_output = self.ffn(query_embeds)
        
        # ✅ RESIDUAL CONNECTION 3
        query_embeds = residual + self.dropout(ffn_output)
        query_embeds = self.norm3(query_embeds)

        return query_embeds