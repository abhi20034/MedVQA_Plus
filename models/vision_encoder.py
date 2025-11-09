# models/vision_encoder.py
import torch.nn as nn
import torchvision.models as tv

class VisionEncoder(nn.Module):
    def __init__(self, proj_dim=768, freeze=True):
        super().__init__()
        # ViT-B/16 from torchvision
        vit = tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1)
        embed_dim = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        self.backbone = vit
        self.proj = nn.Linear(embed_dim, proj_dim)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, img):
        x = self.backbone(img)      # [B, embed_dim]
        x = self.proj(x)            # [B, proj_dim]
        return x
