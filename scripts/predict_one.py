# scripts/predict_one.py
import os, sys
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
import torch

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

def load_label_maps(data_root):
    import json
    maps_path = os.path.join(data_root, "VQA_RAD", "label_maps.json")
    if os.path.exists(maps_path):
        m = json.load(open(maps_path))
        a2i = m["answer2id"]
        # ensure int keys for id2answer
        i2a = {int(k): v for k, v in m["id2answer"].items()}
        return a2i, i2a
    # fallback if only forward map exists
    a2i = json.load(open(os.path.join(data_root, "VQA_RAD", "answer2id.json")))
    i2a = {v: k for k, v in a2i.items()}
    return a2i, i2a

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--ckpt", default="checkpoints/baseline.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--txt_model", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--alpha_tau", type=float, default=2.0)
    args = ap.parse_args()

    # labels
    answer2id, id2answer = load_label_maps(args.data_root)
    num_classes = len(answer2id)

    # models
    vision = VisionEncoder(proj_dim=768, freeze=True).to(device)
    text   = TextEncoder(model_name=args.txt_model, proj_dim=768, freeze=True).to(device)
    fusion = FiLMFusion(dim=768, hidden=512).to(device)
    head   = DecisionScaler(dim=768, num_classes=num_classes, tau=args.alpha_tau).to(device)

    # load weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vision.load_state_dict(ckpt["vision"])
    text.load_state_dict(ckpt["textenc"])
    fusion.load_state_dict(ckpt["fusion"])
    head.load_state_dict(ckpt["head"], strict=False)  # tolerate minor diffs

    vision.eval(); text.eval(); fusion.eval(); head.eval()

    # image preprocessing (match training)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    img = Image.open(args.image).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    tok = AutoTokenizer.from_pretrained(args.txt_model, use_fast=True)
    q = tok(args.question, padding="max_length", truncation=True, max_length=48, return_tensors="pt")
    ids = q["input_ids"].to(device)
    mask= q["attention_mask"].to(device)

    with torch.no_grad():
        e_img = vision(img)
        e_txt = text(ids, mask)
        z_img = fusion(e_img, e_txt)
        logits, alpha, _ = head(z_img, e_txt)
        pred_id = int(logits.argmax(-1).item())
        pred_txt = id2answer.get(pred_id, "<unk>")
        a_img, a_txt = alpha[0].tolist()

    print("Image:", args.image)
    print("Question:", args.question)
    print("Predicted:", pred_txt)
    print("alpha (img, txt):", [round(a_img, 3), round(a_txt, 3)])

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
