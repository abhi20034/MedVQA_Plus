# models/vision_encoder.py
import torch.nn as nn
import torchvision.models as tv

class VisionEncoder(nn.Module):
    def __init__(self, proj_dim=768, freeze=True, dropout=0.1):
        super().__init__()
        # ViT-B/16 from torchvision
        vit = tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1)
        embed_dim = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        self.backbone = vit
        # Use Sequential for new checkpoints, but support loading old checkpoints
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.Dropout(dropout)  # Regularization
        )
        self.dropout_enabled = dropout > 0
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def _load_state_dict_compat(self, state_dict):
        """Handle backward compatibility with old checkpoints."""
        # Check if we have old format (proj.weight) vs new format (proj.0.weight)
        if "proj.weight" in state_dict and "proj.0.weight" not in state_dict:
            # Old checkpoint: single Linear layer
            # Map proj.weight -> proj.0.weight, proj.bias -> proj.0.bias
            new_state_dict = {}
            for key, value in state_dict.items():
                if key == "proj.weight":
                    new_state_dict["proj.0.weight"] = value
                elif key == "proj.bias":
                    new_state_dict["proj.0.bias"] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        return state_dict

    def forward(self, img):
        x = self.backbone(img)      # [B, embed_dim]
        x = self.proj(x)            # [B, proj_dim]
        return x
    
    def load_state_dict(self, state_dict, strict=True):
        """Override to handle backward compatibility."""
        compatible_state_dict = self._load_state_dict_compat(state_dict)
        return super().load_state_dict(compatible_state_dict, strict=strict)
