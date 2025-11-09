# evaluate.py
import os, sys, json, torch
from collections import Counter

# allow imports from project root
sys.path.append(os.path.dirname(__file__))

from dataset_loader import build_dataloaders
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FiLMFusion
from models.decision_scale import DecisionScaler

# ---------- Device ----------
device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("device:", device)
if device == "mps":
    torch.set_float32_matmul_precision("high")
# ----------------------------

@torch.no_grad()
def eval_once(parts, loader, id2answer):
    vision, textenc, fusion, head = parts
    vision.eval(); textenc.eval(); fusion.eval(); head.eval()

    total = 0
    correct = 0
    y_true, y_pred = [], []
    sample_printed = False

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

        pred = logits.argmax(-1)

        correct += (pred[valid] == y[valid]).sum().item()
        total   += valid.sum().item()

        y_true.extend(y[valid].cpu().tolist())
        y_pred.extend(pred[valid].cpu().tolist())

        if not sample_printed:
            k = min(5, img.size(0))
            txt_preds = [id2answer.get(int(p), "<unk>") for p in pred[:k].cpu().tolist()]
            print(">>> sample predictions:", txt_preds)
            if alpha is not None and alpha.ndim == 2 and alpha.size(1) == 2:
                print(">>> sample alpha (img, txt):", [round(x,3) for x in alpha[0].tolist()])
            sample_printed = True

    acc = correct / max(total, 1)
    return acc, y_true, y_pred

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--ckpt", default="checkpoints/baseline.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--txt_model", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--alpha_tau", type=float, default=2.0)
    ap.add_argument("--closed_only", action="store_true")
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    # data & label maps (must match train flags)
    train_loader, val_loader, test_loader, answer2id = build_dataloaders(
        args.data_root, args.txt_model,
        batch=args.batch, num_workers=args.num_workers,
        closed_only=args.closed_only, top_k=args.top_k
    )
    id2answer = {v:k for k, v in answer2id.items()}

    # models
    vision = VisionEncoder(proj_dim=768, freeze=True).to(device)
    text   = TextEncoder(model_name=args.txt_model, proj_dim=768, freeze=True).to(device)
    fusion = FiLMFusion(dim=768, hidden=512).to(device)
    head   = DecisionScaler(dim=768, num_classes=len(answer2id), hidden=256, tau=args.alpha_tau).to(device)

    # load weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vision.load_state_dict(ckpt["vision"])
    text.load_state_dict(ckpt["textenc"])
    fusion.load_state_dict(ckpt["fusion"])
    head.load_state_dict(ckpt["head"], strict=False)  # ok if tau/hidden saved differently

    acc, y_true, y_pred = eval_once((vision, text, fusion, head), test_loader, id2answer)
    print(f">>> test accuracy: {acc:.3f}")

    # compact top-20 report
    from sklearn.metrics import classification_report
    counts = Counter(y_true)
    top = [cls for cls,_ in counts.most_common(20)]
    idx = [i for i,yt in enumerate(y_true) if yt in top]
    yt_top = [y_true[i] for i in idx]
    yp_top = [y_pred[i] for i in idx]
    if len(yt_top) > 0:
        target_names = [f"{c}:{id2answer[c]}" for c in top]
        print(">>> top-20 classes by support (compact report):")
        print(classification_report(
            yt_top, yp_top,
            labels=top,
            target_names=target_names,
            zero_division=0
        ))
    else:
        print(">>> not enough samples to build compact report.")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
