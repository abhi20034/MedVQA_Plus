import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", proj_dim=768, freeze=True):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        hidden = self.lm.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim)
        if freeze:
            for p in self.lm.parameters():
                p.requires_grad = False
    def forward(self, input_ids, attention_mask):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            last = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.proj(pooled)
