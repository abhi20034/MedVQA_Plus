# train.py
import os, json, argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from dataset_loader import build_dataloaders
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FiLMFusion
from models.cross_attention_fusion import CrossAttentionFusion
from models.decision_scale import DecisionScaler

# ---------------- Device selection ----------------
device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(">>> device:", device, flush=True)
if device == "mps":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
# --------------------------------------------------

@torch.no_grad()
def evaluate(parts, loader, id2answer=None, preview_k=3):
    vision, textenc, fusion, head = parts
    vision.eval(); textenc.eval(); fusion.eval(); head.eval()
    correct = total = 0
    preview_done = False

    for batch in loader:
        img  = batch["image"].to(device, non_blocking=False)
        ids  = batch["input_ids"].to(device, non_blocking=False)
        mask = batch["attention_mask"].to(device, non_blocking=False)
        y    = batch["label"].to(device, non_blocking=False)

        valid = y >= 0
        if valid.sum() == 0:
            continue

        e_img = vision(img)
        e_txt = textenc(ids, mask)
        z_img = fusion(e_img, e_txt)
        logits, alpha, _ = head(z_img, e_txt)
        pred  = logits.argmax(dim=-1)

        correct += (pred[valid] == y[valid]).sum().item()
        total   += valid.sum().item()

        if (not preview_done) and id2answer is not None:
            pv = min(preview_k, img.size(0))
            pred_txt = [id2answer.get(int(p.item()), "<unk>") for p in pred[:pv]]
            print(f">>> preview preds (text): {pred_txt}")
            if alpha is not None and alpha.ndim == 2 and alpha.size(1) == 2:
                a = [round(x, 3) for x in alpha[0].tolist()]
                print(">>> preview alpha (img, txt):", a)
            preview_done = True

    return correct / max(total, 1)

def build_class_weights(train_loader, answer2id):
    """Inverse-frequency class weights, normalized to mean≈1."""
    ds = train_loader.dataset
    labels = None
    if hasattr(ds, "items"):
        try:
            labels = [int(it["label"]) for it in ds.items if int(it["label"]) >= 0]
        except Exception:
            labels = None
    if labels is None:
        labels = []
        for i in range(len(ds)):
            try:
                y = int(ds[i]["label"])
            except Exception:
                continue
            if y >= 0:
                labels.append(y)

    counts = Counter(labels)
    K = len(answer2id)
    freq = np.array([counts.get(i, 0) for i in range(K)], dtype=np.float32)
    class_w = 1.0 / np.maximum(freq, 1.0)
    class_w = class_w / class_w.sum() * K
    weights = torch.tensor(class_w, dtype=torch.float32, device=device)
    print(">>> built class weights. nonzero classes:", int((freq > 0).sum()))
    return weights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",   type=str,  default="data")
    ap.add_argument("--epochs",      type=int,  default=12)
    ap.add_argument("--batch",       type=int,  default=8)
    ap.add_argument("--lr",          type=float,default=5e-4)
    ap.add_argument("--txt_model",   type=str,  default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--freeze",      action="store_true", help="Freeze encoders")
    ap.add_argument("--num_workers", type=int,  default=0)
    ap.add_argument("--save",        type=str,  default="checkpoints/baseline.pt")
    ap.add_argument("--max_steps",   type=int,  default=0,  help="Stop after N train steps (0=full epoch)")
    # α-gate controls
    ap.add_argument("--alpha_tau",     type=float, default=2.0,  help="gate softmax temperature (>1 = softer)")
    ap.add_argument("--alpha_entropy", type=float, default=0.02, help="entropy regularizer weight for gate")
    # dataset shaping
    ap.add_argument("--closed_only", action="store_true", help="use only CLOSED yes/no questions")
    ap.add_argument("--top_k", type=int, default=0, help="keep only top-K most frequent answers (0=all)")
    # Loss improvements
    ap.add_argument("--focal_loss", action="store_true", help="Use Focal Loss instead of CE")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (focusing parameter)")
    ap.add_argument("--label_smooth", type=float, default=0.0, help="Label smoothing coefficient (0=no smoothing)")
    # LR scheduling
    ap.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine", "step"], help="LR scheduler type")
    ap.add_argument("--warmup_epochs", type=int, default=1, help="Warmup epochs before cosine decay")
    # Fusion architecture
    ap.add_argument("--fusion_type", type=str, default="film", choices=["film", "cross_attn"], help="Fusion type: film or cross_attn")
    args = ap.parse_args()

    print(">>> args:", args, flush=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Dataset paths sanity
    qa_path = os.path.join(args.data_root, "VQA_RAD", "qa_pairs.json")
    img_dir = os.path.join(args.data_root, "VQA_RAD", "images")
    print(">>> dataset files:",
          f"qa_pairs.json exists? {os.path.exists(qa_path)} | {qa_path}",
          f"images dir exists?   {os.path.isdir(img_dir)} | {img_dir}",
          sep="\n", flush=True)

    # ---- Data ----
    print(">>> building dataloaders ...", flush=True)
    train_loader, val_loader, test_loader, answer2id = build_dataloaders(
        args.data_root, args.txt_model,
        batch=args.batch, num_workers=args.num_workers,
        closed_only=args.closed_only, top_k=args.top_k
    )
    num_classes = len(answer2id)
    print(f">>> answers: {num_classes}", flush=True)

    id2answer = {v: k for k, v in answer2id.items()}
    maps_path = os.path.join(args.data_root, "VQA_RAD", "label_maps.json")
    with open(maps_path, "w") as f:
        json.dump({"answer2id": answer2id, "id2answer": id2answer}, f, indent=2)
    print(f">>> wrote label maps → {maps_path}", flush=True)
    print(">>> sample id2answer:", list(id2answer.items())[:5], flush=True)

    # ---- Models ----
    print(">>> initializing models ...", flush=True)
    vision = VisionEncoder(proj_dim=768, freeze=args.freeze).to(device)
    textenc= TextEncoder(model_name=args.txt_model, proj_dim=768, freeze=args.freeze).to(device)
    
    # Choose fusion type
    if args.fusion_type == "cross_attn":
        fusion = CrossAttentionFusion(dim=768, num_heads=8, dropout=0.1).to(device)
        print(">>> using CrossAttentionFusion")
    else:
        fusion = FiLMFusion(dim=768, hidden=512).to(device)
        print(">>> using FiLMFusion")
    
    head   = DecisionScaler(dim=768, num_classes=num_classes).to(device)
    # set gate temperature from CLI
    if hasattr(head, "tau"):
        head.tau = float(args.alpha_tau)

    # ---- Optimizer (encoders slower LR) ----
    head_params   = [p for p in head.parameters() if p.requires_grad]
    fusion_params = [p for p in fusion.parameters() if p.requires_grad]
    enc_params    = []
    if not args.freeze:
        enc_params += [p for p in vision.parameters() if p.requires_grad]
        enc_params += [p for p in textenc.parameters() if p.requires_grad]

    opt = AdamW([
        {"params": enc_params,    "lr": args.lr * 0.2},  # encoders: smaller LR
        {"params": fusion_params, "lr": args.lr},
        {"params": head_params,   "lr": args.lr},
    ])
    
    # ---- Learning Rate Scheduler ----
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
        print(f">>> using CosineAnnealingLR scheduler (T_max={args.epochs})")
    elif args.lr_scheduler == "step":
        scheduler = StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)
        print(f">>> using StepLR scheduler (step_size={max(1, args.epochs // 3)}, gamma=0.5)")
    else:
        scheduler = None
        print(">>> no LR scheduler")

    # ---- Loss (class-weighted) ----
    class_weights = build_class_weights(train_loader, answer2id)
    
    if args.focal_loss:
        # Focal Loss implementation for class imbalance
        class FocalLoss(nn.Module):
            def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
                super().__init__()
                self.weight = weight
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(
                    inputs, targets, weight=self.weight, 
                    label_smoothing=self.label_smoothing, reduction='none'
                )
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                return focal_loss.mean()
        
        crit = FocalLoss(weight=class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smooth)
        print(f">>> using Focal Loss (gamma={args.focal_gamma}, label_smoothing={args.label_smooth})")
    else:
        crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smooth)
        if args.label_smooth > 0:
            print(f">>> using CrossEntropyLoss with label smoothing={args.label_smooth}")

    # ---- Best checkpoint tracking ----
    # Derive best checkpoint path from --save argument
    # e.g., "checkpoints/test.pt" → "checkpoints/best_test.pt"
    save_dir = os.path.dirname(args.save) or "checkpoints"
    save_basename = os.path.basename(args.save)
    # Remove .pt extension if present
    if save_basename.endswith('.pt'):
        save_basename = save_basename[:-3]
    best_path = os.path.join(save_dir, f"best_{save_basename}.pt")
    best_val = -1.0
    final_val_acc = 0.0  # Initialize for final checkpoint
    print(f">>> checkpoint paths: best → {best_path}, final → {args.save}", flush=True)

    # ---- Training ----
    for epoch in range(1, args.epochs + 1):
        print(f">>> starting epoch {epoch}", flush=True)
        vision.train(); textenc.train(); fusion.train(); head.train()
        running_loss, steps = 0.0, 0

        for batch in train_loader:
            img  = batch["image"].to(device, non_blocking=False)
            ids  = batch["input_ids"].to(device, non_blocking=False)
            mask = batch["attention_mask"].to(device, non_blocking=False)
            y    = batch["label"].to(device, non_blocking=False)

            valid = y >= 0
            if valid.sum() == 0:
                continue

            # forward
            e_img = vision(img)
            e_txt = textenc(ids, mask)
            z_img = fusion(e_img, e_txt)
            logits, alpha, _ = head(z_img, e_txt)

            # main CE loss
            loss = crit(logits[valid], y[valid])

            # ---- α entropy regularizer (discourage 0/1 gate) ----
            # Target uniform distribution: H_max = log(2) ≈ 0.693
            # We want to maximize entropy (penalize low entropy)
            alpha_safe = alpha.clamp_min(1e-6)
            entropy = -(alpha_safe * alpha_safe.log()).sum(1)     # H(α) per sample
            max_entropy = torch.log(torch.tensor(2.0, device=device))  # log(2) for uniform
            # Penalty: encourage entropy to be close to max_entropy
            entropy_loss = (max_entropy - entropy.mean()).clamp(min=0.0)
            alpha_reg = args.alpha_entropy * entropy_loss
            loss = loss + alpha_reg
            # ------------------------------------------------------

            # backward
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # gradient clipping
            all_params = []
            for m in (vision, textenc, fusion, head):
                all_params += [p for p in m.parameters() if p.requires_grad and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            opt.step()

            steps += 1
            running_loss += loss.item()
            print(f"epoch {epoch} step {steps} loss={loss.item():.4f}", flush=True)

            # alpha logging every 20 steps
            if steps % 20 == 0 and alpha is not None and alpha.ndim == 2 and alpha.size(1) == 2:
                a_mean = alpha.detach().mean(0).tolist()
                print(f"alpha mean (img, txt): {a_mean}", flush=True)

            if args.max_steps and steps >= args.max_steps:
                print(f">>> reached max_steps={args.max_steps}, stopping epoch early.", flush=True)
                break

        # ---- Validation ----
        val_acc = evaluate((vision, textenc, fusion, head), val_loader, id2answer=id2answer)
        print(f">>> epoch {epoch}/{args.epochs} | train_loss={running_loss/max(steps,1):.4f} | val_acc={val_acc:.3f}", flush=True)
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = opt.param_groups[0]['lr']
            print(f">>> LR updated to {current_lr:.6f}", flush=True)

        # save best on val
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "vision": vision.state_dict(),
                "textenc": textenc.state_dict(),
                "fusion": fusion.state_dict(),
                "head":   head.state_dict(),
                "answer2id": answer2id,
                "id2answer": id2answer,
                "epoch": epoch,
                "val_acc": float(val_acc),
                "checkpoint_type": "best"
            }, best_path)
            print(f">>> new best val_acc={best_val:.3f} (epoch {epoch}) → saved {best_path}", flush=True)
        
        # Store last epoch's val_acc for final checkpoint
        final_val_acc = val_acc

    # ---- Save final checkpoint (last epoch) ----
    torch.save({
        "vision": vision.state_dict(),
        "textenc": textenc.state_dict(),
        "fusion": fusion.state_dict(),
        "head":   head.state_dict(),
        "answer2id": answer2id,
        "id2answer": id2answer,
        "epoch": args.epochs,
        "val_acc": float(final_val_acc),
        "best_val_acc": float(best_val),
        "checkpoint_type": "final"
    }, args.save)
    print(f">>> saved final checkpoint (epoch {args.epochs}, val_acc={final_val_acc:.3f}, best={best_val:.3f}) → {args.save}", flush=True)

    # ---- Test on last-epoch weights (for reference) ----
    test_acc = evaluate((vision, textenc, fusion, head), test_loader, id2answer=id2answer)
    print(f">>> test_acc (last epoch) = {test_acc:.3f}", flush=True)
    print(">>> done.", flush=True)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()