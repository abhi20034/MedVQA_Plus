#!/usr/bin/env python3
import os, json, argparse, random
from collections import defaultdict
random.seed(42)

def canon_ans(a):
    a = str(a).strip().lower()
    if a in {"y", "yes", "true"}:
        return "yes"
    if a in {"n", "no", "false"}:
        return "no"
    return a

def stratified_split(items, val_frac=0.1, test_frac=0.1):
    buckets = defaultdict(list)
    for i, it in enumerate(items):
        buckets[it.get("type", "UNKNOWN")].append(i)
    tr, va, te = set(), set(), set()
    for _, idxs in buckets.items():
        random.shuffle(idxs)
        n = len(idxs); nv = int(n*val_frac); nt = int(n*test_frac)
        va.update(idxs[:nv]); te.update(idxs[nv:nv+nt]); tr.update(idxs[nv+nt:])
    return tr, va, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    images_set = set(os.listdir(args.images_dir))
    std = []
    skipped = 0

    for it in raw:
        img = (it.get("image_name") or "").strip()
        q   = (it.get("question") or "").strip()
        a   = canon_ans(it.get("answer"))
        t   = (it.get("answer_type") or "").strip().upper() or None
        if not t:
            t = "CLOSED" if a in {"yes","no"} else "OPEN"

        if img and img not in images_set:
            for ext in (".jpg", ".png", ".jpeg"):
                if img+ext in images_set:
                    img = img+ext; break

        if not (img and q and a):
            skipped += 1; continue

        std.append({"image_id": img, "question": q, "answer": a, "type": t})

    print(f"Converted {len(std)} entries; skipped {skipped} missing fields.")

    tr, va, te = stratified_split(std, args.val_frac, args.test_frac)
    for i, x in enumerate(std):
        x["split"] = "train" if i in tr else ("val" if i in va else "test")
    print(f"Splits: train={len(tr)} val={len(va)} test={len(te)}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(std, f, indent=2)
    print(f"Wrote standardized JSON → {args.out_json}")

if __name__ == "__main__":
    main()