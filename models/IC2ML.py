import torch
import torch.nn as nn
import math

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels//4)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels//4)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm2d(out_channels//4)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels//4)
        )
        
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
class CrossAttention(nn.Module):
    def __init__(self, d_ff):
        super().__init__()
        self.query = nn.Linear(d_ff, d_ff)
        self.key = nn.Linear(d_ff, d_ff)
        self.value = nn.Linear(d_ff, d_ff)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = d_ff ** -0.5
        
    def forward(self, context_features, cnn_features):
        q = self.query(context_features).unsqueeze(1)  
        k = self.key(cnn_features).unsqueeze(1)       
        v = self.value(cnn_features).unsqueeze(1)   
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  
        attn_probs = self.softmax(attn_scores)
        attended = torch.bmm(attn_probs, v).squeeze(1)  
        return attended
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.context = args.context
        self.horizon = args.horizon
        self.intra_embedding = nn.Sequential(
            nn.Linear(args.charge_discharge_length//2, args.d_ff//2),
            nn.GELU(),
            nn.LayerNorm(args.d_ff//2),
            nn.Linear(args.d_ff//2, args.d_ff),
            nn.LayerNorm(args.d_ff)
        )
        self.soh_predictor = nn.Sequential(
            nn.Linear(args.d_ff, args.d_ff//2),
            nn.GELU(),
            nn.LayerNorm(args.d_ff//2),
            nn.Linear(args.d_ff//2, 1)
        )
        
        self.intercycle_attention = nn.MultiheadAttention(
            embed_dim=args.d_ff,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )
        self.pos_embedding = PositionalEncoding(args.d_ff, dropout=0.1)
        self.norm1 = nn.LayerNorm(args.d_ff)
        self.norm2 = nn.LayerNorm(args.d_ff)
        
        self.flatten_processor = nn.Sequential(
            nn.Linear(args.context * args.d_ff, args.d_ff),
            nn.GELU(),
            nn.LayerNorm(args.d_ff)
        )

        self.intraintercycle_embedding = nn.Sequential(
            InceptionBlock(1, 64), 
            nn.MaxPool2d(kernel_size=2, stride=1),  
            InceptionBlock(64, 128), 
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.cnn_projection = nn.Linear(128, args.d_ff)
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(args.d_ff, args.d_ff),
            nn.GELU(),
            nn.LayerNorm(args.d_ff),
            nn.Linear(args.d_ff, args.horizon)
        )

        self.cross_attention = CrossAttention(args.d_ff)
        self.RUL_predictor = nn.Sequential(
            nn.Linear(args.d_ff, args.d_ff//2),  
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(args.d_ff//2, 1),
        )

    def forward(self, capacity_increment, capacity_mask, tgt_soh=None):
        batch_size = capacity_increment.size(0)
        x = self.intra_embedding(capacity_increment)  
        soh = self.soh_predictor(x).squeeze(-1)
        
        x = self.pos_embedding(x)
        attn_output, _ = self.intercycle_attention(query=x, key=x, value=x)
        x = self.norm1(x + attn_output)  
        flattened = x.reshape(batch_size, -1)  
        context_features = self.flatten_processor(flattened) 

        cnn_input = capacity_increment.unsqueeze(1)  
        cnn_features = self.intraintercycle_embedding(cnn_input) 
        cnn_features = cnn_features.squeeze(-1).squeeze(-1)  
        cnn_features = self.cnn_projection(cnn_features)  
        
        attended_features = self.cross_attention(context_features, cnn_features)
        trajectory = self.trajectory_predictor(context_features)
        RUL = self.RUL_predictor(attended_features).squeeze(-1) 
        
        return soh, trajectory, RUL