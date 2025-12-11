# models/cross_attention_fusion.py
import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between vision and text features.
    Vision attends to text and vice versa, then both are fused.
    More sophisticated than FiLM for complex interactions.
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # Cross-attention: vision queries text keys/values
        self.v_to_t_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        # Cross-attention: text queries vision keys/values
        self.t_to_v_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer norms
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_fused = nn.LayerNorm(dim)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, v_feat, t_feat):
        """
        v_feat: [B, D] vision features
        t_feat: [B, D] text features
        returns: [B, D] fused features
        """
        # Expand to sequence dimension for attention
        v_seq = v_feat.unsqueeze(1)  # [B, 1, D]
        t_seq = t_feat.unsqueeze(1)  # [B, 1, D]
        
        # Vision attends to text
        v_attended, _ = self.v_to_t_attn(v_seq, t_seq, t_seq)  # [B, 1, D]
        v_attended = self.norm_v(v_attended.squeeze(1) + v_feat)  # [B, D]
        
        # Text attends to vision
        t_attended, _ = self.t_to_v_attn(t_seq, v_seq, v_seq)  # [B, 1, D]
        t_attended = self.norm_t(t_attended.squeeze(1) + t_feat)  # [B, D]
        
        # Fuse both attended features
        fused = torch.cat([v_attended, t_attended], dim=-1)  # [B, 2D]
        fused = self.fusion(fused)  # [B, D]
        fused = self.norm_fused(fused)
        
        return fused

