# scripts/sanity_step.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

import torch
from dataset_loader import build_dataloaders
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FiLMFusion
from models.decision_scale import DecisionScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

train_loader, _, _, a2i = build_dataloaders("data", "emilyalsentzer/Bio_ClinicalBERT", batch=2, num_workers=0)
print("num_classes:", len(a2i))

vision = VisionEncoder().to(device)
text   = TextEncoder().to(device)
fusion = FiLMFusion().to(device)
head   = DecisionScaler(num_classes=len(a2i)).to(device)

batch = next(iter(train_loader))
img  = batch["image"].to(device)
ids  = batch["input_ids"].to(device)
mask = batch["attention_mask"].to(device)
y    = batch["label"].to(device)
print("batch shapes:", img.shape, ids.shape, mask.shape, y.shape)

with torch.no_grad():
    v = vision(img)
    t = text(ids, mask)
    z = fusion(v, t)
    logits, alpha, _ = head(z, t)

print("logits:", logits.shape, "alpha:", alpha.shape)
print("alpha[0]:", alpha[0].tolist())
print("OK")
