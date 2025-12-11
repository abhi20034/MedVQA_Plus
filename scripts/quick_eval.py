"""Lightweight eval harness for MedVQA+ classifiers.

Expected data format: JSONL with fields:
  - image: path to image file
  - question: question text
  - answer: ground truth answer string (optional; if absent, just runs inference)
  - modality: optional hint (chest/neuro/abdomen/auto)

This script measures closed/open accuracy (string match) and gate stats.
Designed for small batches on Mac (MPS), batch size=1.
"""
import argparse
import json
import os
import re
from typing import Dict, List, Optional

import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from dataset_loader import CLIP_MEAN, CLIP_STD
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FiLMFusion
from models.cross_attention_fusion import CrossAttentionFusion
from models.decision_scale import DecisionScaler

# Question routing patterns (kept in sync with app.py)
CLOSED_PATTERNS = [
    r"\b(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\s+",
    r"\b(is\s+there|are\s+there|does\s+it|do\s+they)\s+",
    r"\?$.*\b(yes|no)\b",
]

OPEN_PATTERNS = [
    r"\b(what|where|when|who|which|why|how)\s+",
    r"\b(describe|identify|name|list|show|indicate)\s+",
]

EXPANDED_OPEN = OPEN_PATTERNS + [
    r"\b(explain|compare|summarize|characterize|estimate|measure)\s+",
    r"\b(location|extent|severity)\b",
]


def is_closed_question(question: str) -> bool:
    q = question.lower().strip()
    for pattern in EXPANDED_OPEN:
        if re.search(pattern, q, re.IGNORECASE):
            return False
    if re.search(r"\b(yes|no)\b", q) and q.endswith("?"):
        if not any(re.search(p, q, re.IGNORECASE) for p in EXPANDED_OPEN):
            return True
    for pattern in CLOSED_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    if re.match(r"^(is|are|does|do|was|were|has|have|can|could|will|would|should)\s+", q):
        return True
    return False


def detect_modality(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ['brain', 'neuro', 'cerebral', 'cerebellum', 'ventricle', 'lesion', 'infarct', 'hemorrhage', 'mri', 'ct head', 'skull', 'cortex']):
        return 'neuro'
    if any(word in q for word in ['chest', 'lung', 'pulmonary', 'pleural', 'heart', 'cardiac', 'thorax', 'mediastinum']):
        return 'chest'
    if any(word in q for word in ['abdomen', 'abdominal', 'liver', 'kidney', 'pancreas', 'gallbladder', 'bowel', 'stomach']):
        return 'abdomen'
    return 'chest'


def build_transform():
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: str, fusion_type: str, txt_model: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    answer2id = ckpt["answer2id"]
    id2answer = ckpt.get("id2answer", {v: k for k, v in answer2id.items()})
    num_classes = len(answer2id)

    vision = VisionEncoder(proj_dim=768, freeze=True).to(device)
    textenc = TextEncoder(model_name=txt_model, proj_dim=768, freeze=True).to(device)
    fusion = CrossAttentionFusion(dim=768, num_heads=8, dropout=0.1).to(device) if fusion_type == "cross_attn" else FiLMFusion(dim=768, hidden=512).to(device)
    head = DecisionScaler(dim=768, num_classes=num_classes).to(device)

    vision.load_state_dict(ckpt["vision"], strict=False)
    textenc.load_state_dict(ckpt["textenc"], strict=False)
    fusion.load_state_dict(ckpt["fusion"], strict=False)
    head.load_state_dict(ckpt["head"], strict=False)

    vision.eval(); textenc.eval(); fusion.eval(); head.eval()
    return vision, textenc, fusion, head, id2answer


def run_eval(data_path: str, closed_ckpt: str, open_ckpt: str, txt_model: str, fusion_open: str, limit: Optional[int]):
    device = get_device()
    transform = build_transform()
    tokenizer = AutoTokenizer.from_pretrained(txt_model, use_fast=True)

    closed_model = load_model(closed_ckpt, fusion_type="film", txt_model=txt_model, device=device)
    open_model = load_model(open_ckpt, fusion_type=fusion_open, txt_model=txt_model, device=device)

    stats = {
        'closed_total': 0,
        'closed_correct': 0,
        'open_total': 0,
        'open_correct': 0,
        'alpha_img_sum': 0.0,
        'alpha_txt_sum': 0.0,
        'n': 0,
    }

    def iter_samples(path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                sample = json.loads(line)
                yield sample

    for sample in iter_samples(data_path):
        image_path = sample.get("image")
        question = sample.get("question", "").strip()
        answer = sample.get("answer")
        modality = sample.get("modality", "auto").lower()

        if not image_path or not question:
            continue
        if modality == "auto":
            modality = detect_modality(question)

        is_closed = is_closed_question(question)
        vision, textenc, fusion, head, id2answer = closed_model if is_closed else open_model

        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        enc = tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            e_img = vision(img_tensor)
            e_txt = textenc(enc["input_ids"], enc["attention_mask"])
            z_img = fusion(e_img, e_txt)
            logits, alpha, _ = head(z_img, e_txt)
            pred_id = int(logits.argmax(dim=-1)[0].item())
            pred_answer = id2answer.get(pred_id, "<unk>")
            alpha_img = float(alpha[0, 0])
            alpha_txt = float(alpha[0, 1])

        stats['alpha_img_sum'] += alpha_img
        stats['alpha_txt_sum'] += alpha_txt
        stats['n'] += 1

        if is_closed:
            stats['closed_total'] += 1
            if answer is not None and pred_answer.lower() == str(answer).lower():
                stats['closed_correct'] += 1
        else:
            stats['open_total'] += 1
            if answer is not None and pred_answer.lower() == str(answer).lower():
                stats['open_correct'] += 1

    def safe_div(num, den):
        return num / den if den else 0.0

    print("=== Quick Eval Summary ===")
    print(f"Device: {device}")
    print(f"Samples evaluated: {stats['n']}")
    print(f"Closed accuracy: {safe_div(stats['closed_correct'], stats['closed_total']):.3f} ({stats['closed_correct']}/{stats['closed_total']})")
    print(f"Open accuracy:   {safe_div(stats['open_correct'], stats['open_total']):.3f} ({stats['open_correct']}/{stats['open_total']})")
    print(f"Gate mean α_img: {safe_div(stats['alpha_img_sum'], stats['n']):.3f}")
    print(f"Gate mean α_txt: {safe_div(stats['alpha_txt_sum'], stats['n']):.3f}")


def parse_args():
    ap = argparse.ArgumentParser(description="Quick eval for MedVQA+ classifiers")
    ap.add_argument("--data", required=True, help="Path to JSONL file")
    ap.add_argument("--closed_ckpt", default="checkpoints/best_closed_improved.pt")
    ap.add_argument("--open_ckpt", default="checkpoints/best_topk50_improved.pt")
    ap.add_argument("--txt_model", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--fusion_open", default="film", choices=["film", "cross_attn"], help="Fusion type for open-ended checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on samples")
    return ap.parse_args()


def main():
    args = parse_args()
    run_eval(
        data_path=args.data,
        closed_ckpt=args.closed_ckpt,
        open_ckpt=args.open_ckpt,
        txt_model=args.txt_model,
        fusion_open=args.fusion_open,
        limit=args.limit,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
