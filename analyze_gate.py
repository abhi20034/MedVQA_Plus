# analyze_gate.py
"""
Utility script to analyze alpha gate behavior across dataset.
Visualizes distribution of gate weights and correlation with accuracy.
"""
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from dataset_loader import build_dataloaders
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FiLMFusion
from models.decision_scale import DecisionScaler

device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f">>> device: {device}")


@torch.no_grad()
def analyze_gate(ckpt_path, data_root, txt_model, closed_only=False, top_k=0, 
                 split="test", max_samples=500):
    """Analyze gate behavior across dataset."""
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    answer2id = ckpt["answer2id"]
    id2answer = {v: k for k, v in answer2id.items()}
    
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_root, txt_model, batch=8, num_workers=0,
        closed_only=closed_only, top_k=top_k
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    
    vision = VisionEncoder(proj_dim=768, freeze=True).to(device)
    textenc = TextEncoder(model_name=txt_model, proj_dim=768, freeze=True).to(device)
    fusion = FiLMFusion(dim=768, hidden=512).to(device)
    head = DecisionScaler(dim=768, num_classes=len(answer2id)).to(device)
    
    vision.load_state_dict(ckpt["vision"])
    textenc.load_state_dict(ckpt["textenc"])
    fusion.load_state_dict(ckpt["fusion"])
    head.load_state_dict(ckpt["head"], strict=False)
    
    vision.eval(); textenc.eval(); fusion.eval(); head.eval()
    
    # Collect statistics
    alpha_imgs = []
    alpha_txts = []
    entropies = []
    corrects = []
    questions = []
    
    count = 0
    for batch in loader:
        if count >= max_samples:
            break
            
        img = batch["image"].to(device, non_blocking=False)
        ids = batch["input_ids"].to(device, non_blocking=False)
        mask = batch["attention_mask"].to(device, non_blocking=False)
        y = batch["label"].to(device, non_blocking=False)
        
        valid = y >= 0
        if valid.sum() == 0:
            continue
        
        e_img = vision(img)
        e_txt = textenc(ids, mask)
        z_img = fusion(e_img, e_txt)
        logits, alpha, _ = head(z_img, e_txt)
        
        pred = logits.argmax(dim=-1)
        is_correct = (pred == y)
        
        # Extract alpha values for valid samples
        alpha_valid = alpha[valid].cpu()
        a_img = alpha_valid[:, 0].tolist()
        a_txt = alpha_valid[:, 1].tolist()
        alpha_imgs.extend(a_img)
        alpha_txts.extend(a_txt)
        
        # Compute entropy
        alpha_safe = alpha_valid.clamp_min(1e-6)
        entropy_batch = -(alpha_safe * alpha_safe.log()).sum(1)  # [B]
        entropies.extend(entropy_batch.cpu().tolist())
        
        # Track correctness
        corrects.extend(is_correct[valid].cpu().tolist())
        
        # Questions for context
        qs = batch.get("question", ["?"] * img.size(0))
        if isinstance(qs, list):
            questions.extend([qs[i] for i, v in enumerate(valid.cpu().tolist()) if v])
        
        count += valid.sum().item()
    
    alpha_imgs = np.array(alpha_imgs)
    alpha_txts = np.array(alpha_txts)
    entropies = np.array(entropies)
    corrects = np.array(corrects)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Gate Analysis: {split} set ({len(alpha_imgs)} samples)")
    print(f"{'='*60}")
    print(f"Alpha Image - Mean: {alpha_imgs.mean():.3f}, Std: {alpha_imgs.std():.3f}")
    print(f"              Min: {alpha_imgs.min():.3f}, Max: {alpha_imgs.max():.3f}")
    print(f"Alpha Text  - Mean: {alpha_txts.mean():.3f}, Std: {alpha_txts.std():.3f}")
    print(f"              Min: {alpha_txts.min():.3f}, Max: {alpha_txts.max():.3f}")
    print(f"Entropy     - Mean: {entropies.mean():.3f}, Std: {entropies.std():.3f}")
    print(f"              Target (uniform): {np.log(2):.3f}")
    print(f"Accuracy    - Overall: {corrects.mean():.3f}")
    print(f"              When img > txt: {corrects[alpha_imgs > alpha_txts].mean():.3f} ({np.sum(alpha_imgs > alpha_txts)} samples)")
    print(f"              When txt > img: {corrects[alpha_txts > alpha_imgs].mean():.3f} ({np.sum(alpha_txts > alpha_imgs)} samples)")
    print(f"{'='*60}\n")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Alpha distribution
    axes[0, 0].hist(alpha_imgs, bins=30, alpha=0.6, label='Image', color='blue')
    axes[0, 0].hist(alpha_txts, bins=30, alpha=0.6, label='Text', color='red')
    axes[0, 0].axvline(0.5, color='black', linestyle='--', linewidth=1, label='Balanced')
    axes[0, 0].set_xlabel('Alpha Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Alpha Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Alpha scatter plot
    axes[0, 1].scatter(alpha_imgs, alpha_txts, alpha=0.5, c=corrects, cmap='RdYlGn')
    axes[0, 1].plot([0, 1], [1, 0], 'k--', linewidth=1, label='Balanced line')
    axes[0, 1].set_xlabel('Alpha Image')
    axes[0, 1].set_ylabel('Alpha Text')
    axes[0, 1].set_title('Alpha Image vs Alpha Text (colored by correctness)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Entropy distribution
    axes[1, 0].hist(entropies, bins=30, color='purple', alpha=0.7)
    axes[1, 0].axvline(np.log(2), color='black', linestyle='--', linewidth=2, label=f'Max entropy ({np.log(2):.3f})')
    axes[1, 0].set_xlabel('Entropy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Gate Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Accuracy by alpha ratio
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    acc_by_alpha = []
    for i in range(len(bins) - 1):
        mask = (alpha_imgs >= bins[i]) & (alpha_imgs < bins[i+1])
        if mask.sum() > 0:
            acc_by_alpha.append(corrects[mask].mean())
        else:
            acc_by_alpha.append(np.nan)
    
    axes[1, 1].bar(bin_centers, acc_by_alpha, width=0.08, color='orange', alpha=0.7)
    axes[1, 1].axhline(corrects.mean(), color='black', linestyle='--', linewidth=1, label=f'Overall acc ({corrects.mean():.3f})')
    axes[1, 1].set_xlabel('Alpha Image')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy by Alpha Image Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = f"gate_analysis_{split}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f">>> Saved visualization → {output_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--txt_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--closed_only", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=500)
    args = ap.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    analyze_gate(
        args.ckpt, args.data_root, args.txt_model,
        closed_only=args.closed_only, top_k=args.top_k,
        split=args.split, max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

