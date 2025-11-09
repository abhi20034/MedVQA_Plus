# dataset_loader.py
import json, os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import AutoTokenizer
from collections import Counter

import os as _os
_os.environ["TOKENIZERS_PARALLELISM"] = "false"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def _canon_yesno(a: str):
    a = str(a or "").strip().lower()
    if a in {"y", "yes", "true", "1"}:  return "yes"
    if a in {"n", "no", "false", "0"}:  return "no"
    return a

class VQARadDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_len=48,
        answer2id=None,
        items=None,
    ):
        self.img_dir = os.path.join(data_root, "VQA_RAD", "images")
        self.qa_path = os.path.join(data_root, "VQA_RAD", "qa_pairs.json")

        if items is None:
            with open(self.qa_path, "r", encoding="utf-8") as f:
                all_items = json.load(f)
            self.items = [x for x in all_items if x.get("split", "train") == split]
        else:
            # pre-filtered view for each split
            self.items = [x for x in items if x.get("split", "train") == split]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

        assert answer2id is not None, "VQARadDataset requires an answer2id mapping"
        self.answer2id = answer2id

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(os.path.join(self.img_dir, it["image_id"])).convert("RGB")
        img = self.transform(img)

        tok = self.tokenizer(
            it["question"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        input_ids = tok["input_ids"].squeeze(0)
        attn_mask = tok["attention_mask"].squeeze(0)

        ans = (it["answer"] or "").strip().lower()
        label = torch.tensor(self.answer2id.get(ans, -1), dtype=torch.long)
        return {
            "image": img,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "label": label,
            "question": it["question"],
            "answer": ans,
            "image_id": it["image_id"],
        }

def build_dataloaders(
    data_root,
    model_name,
    batch=32,
    num_workers=2,
    closed_only=False,
    top_k=0,
):
    """Build DataLoaders with optional filtering:
       - closed_only=True  → keep only yes/no items, vocab=['no','yes']
       - top_k>0           → keep only top_k frequent answers (multi-class)

       Returns: train_loader, val_loader, test_loader, answer2id
    """
    qa_path = os.path.join(data_root, "VQA_RAD", "qa_pairs.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Optional CLOSED-only filter (yes/no)
    if closed_only:
        for it in items:
            it["answer"] = _canon_yesno(it.get("answer"))
        items = [it for it in items if it.get("type", "CLOSED").upper() == "CLOSED"]
        # force vocabulary order to be stable
        answer2id = {"no": 0, "yes": 1}
        # drop anything not yes/no after canon
        items = [it for it in items if it["answer"] in answer2id]
    else:
        # Optional top_k filter for multi-class
        if top_k and top_k > 0:
            counts = Counter([(it.get("answer") or "").strip().lower() for it in items])
            keep = set([a for a, _ in counts.most_common(top_k)])
            items = [it for it in items if (it.get("answer") or "").strip().lower() in keep]

        # Build vocab from WHAT REMAINS
        answers = [(it.get("answer") or "").strip().lower() for it in items]
        counts = Counter(answers)
        vocab = [a for a, _ in counts.most_common()]
        answer2id = {a: i for i, a in enumerate(vocab)}

    # Construct per-split datasets that share the same vocab
    train_ds = VQARadDataset(data_root, "train", model_name=model_name, answer2id=answer2id, items=items)
    val_ds   = VQARadDataset(data_root, "val",   model_name=model_name, answer2id=answer2id, items=items)
    test_ds  = VQARadDataset(data_root, "test",  model_name=model_name, answer2id=answer2id, items=items)

    kw = dict(batch_size=batch, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds, shuffle=False, **kw)


    return train_loader, val_loader, test_loader, answer2id
