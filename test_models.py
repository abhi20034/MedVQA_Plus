# test_models.py
"""
Comprehensive test script for both closed and top-k models.
Tests multiple examples and calculates accuracy.
Uses semantic similarity matching for top-k model.
"""
import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from dataset_loader import CLIP_MEAN, CLIP_STD
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FiLMFusion
from models.decision_scale import DecisionScaler
from semantic_eval import SemanticMatcher

device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

def load_model(ckpt_path, fusion_type="film"):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    answer2id = ckpt["answer2id"]
    id2answer = {v: k for k, v in answer2id.items()}
    num_classes = len(answer2id)
    
    vision = VisionEncoder(proj_dim=768, freeze=True).to(device)
    textenc = TextEncoder(model_name="emilyalsentzer/Bio_ClinicalBERT", proj_dim=768, freeze=True).to(device)
    
    if fusion_type == "cross_attn":
        from models.cross_attention_fusion import CrossAttentionFusion
        fusion = CrossAttentionFusion(dim=768, num_heads=8, dropout=0.1).to(device)
    else:
        fusion = FiLMFusion(dim=768, hidden=512).to(device)
    
    head = DecisionScaler(dim=768, num_classes=num_classes).to(device)
    
    vision.load_state_dict(ckpt["vision"], strict=False)
    textenc.load_state_dict(ckpt["textenc"], strict=False)
    fusion.load_state_dict(ckpt["fusion"], strict=False)
    head.load_state_dict(ckpt["head"], strict=False)
    
    vision.eval()
    textenc.eval()
    fusion.eval()
    head.eval()
    
    return vision, textenc, fusion, head, id2answer, answer2id

def test_examples(ckpt_path, examples, model_type="closed", fusion_type="film", use_semantic=False):
    """Test model on multiple examples and calculate accuracy.
    
    Args:
        use_semantic: If True, use semantic similarity matching (for top-k model)
    """
    print(f"\n{'='*70}")
    print(f"Testing {model_type.upper()} Model: {ckpt_path}")
    if use_semantic:
        print("Using SEMANTIC SIMILARITY matching (threshold=0.7)")
    print(f"{'='*70}\n")
    
    vision, textenc, fusion, head, id2answer, answer2id = load_model(ckpt_path, fusion_type)
    
    # Initialize semantic matcher if needed
    semantic_matcher = None
    if use_semantic:
        semantic_matcher = SemanticMatcher(device=device)
        print("Initialized semantic similarity matcher...\n")
    
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
    img_dir = "data/VQA_RAD/images"
    
    correct = 0
    total = 0
    results = []
    alpha_imgs = []
    alpha_txts = []
    
    for i, ex in enumerate(examples, 1):
        img_path = os.path.join(img_dir, ex["image_id"])
        if not os.path.exists(img_path):
            continue
        
        question = ex["question"]
        true_answer = (ex.get("answer") or "").strip().lower()
        
        # Preprocess
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        enc = tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            e_img = vision(img_tensor)
            e_txt = textenc(enc["input_ids"], enc["attention_mask"])
            z_img = fusion(e_img, e_txt)
            logits, alpha, _ = head(z_img, e_txt)
            
            pred_id = int(logits.argmax(dim=-1)[0].item())
            pred_answer = id2answer.get(pred_id, "<unk>").lower()
            
            alpha_img = float(alpha[0, 0])
            alpha_txt = float(alpha[0, 1])
            alpha_imgs.append(alpha_img)
            alpha_txts.append(alpha_txt)
        
        # Check correctness
        if use_semantic and semantic_matcher:
            # Use semantic similarity matching
            # Get all vocabulary answers as candidates
            vocab_answers = list(answer2id.keys())
            best_match, sim_score = semantic_matcher.best_match(pred_answer, vocab_answers, threshold=0.7)
            
            # Check if true answer is semantically similar to prediction
            # First check exact match
            is_exact = (pred_answer == true_answer)
            
            # Then check semantic similarity
            if not is_exact:
                true_sim = semantic_matcher.similarity(pred_answer, true_answer)
                is_semantic = (true_sim >= 0.7)  # Threshold for semantic match
            else:
                true_sim = 1.0
                is_semantic = True
            
            is_correct = is_exact or is_semantic
            match_type = "exact" if is_exact else ("semantic" if is_semantic else "none")
            
        else:
            # Exact string matching
            is_correct = (pred_answer == true_answer)
            match_type = "exact" if is_correct else "none"
            true_sim = 1.0 if is_correct else 0.0
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        results.append({
            "question": question,
            "true": true_answer,
            "pred": pred_answer,
            "correct": is_correct,
            "match_type": match_type,
            "similarity": true_sim,
            "alpha_img": alpha_img,
            "alpha_txt": alpha_txt,
        })
        
        sim_str = f" | sim={true_sim:.2f}" if use_semantic else ""
        print(f"{status} [{i}/{len(examples)}] Q: {question[:50]}...")
        print(f"   True: {true_answer:25s} | Pred: {pred_answer:25s} | {match_type}{sim_str} | α: [{alpha_img:.3f}, {alpha_txt:.3f}]")
    
    accuracy = correct / total if total > 0 else 0.0
    avg_alpha_img = sum(alpha_imgs) / len(alpha_imgs) if alpha_imgs else 0.0
    avg_alpha_txt = sum(alpha_txts) / len(alpha_txts) if alpha_txts else 0.0
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"  Average Gate: α_img={avg_alpha_img:.3f}, α_txt={avg_alpha_txt:.3f}")
    print(f"{'='*70}\n")
    
    return accuracy, results, alpha_imgs, alpha_txts

def main():
    # Load test examples
    with open("data/VQA_RAD/qa_pairs.json", "r") as f:
        all_data = json.load(f)
    
    # Closed examples (yes/no)
    closed_examples = [
        x for x in all_data 
        if x.get("type", "").upper() == "CLOSED" 
        and x.get("split") == "test"
        and (x.get("answer") or "").strip().lower() in {"yes", "no"}
    ][:20]  # Test 20 examples
    
    # Open examples (top-k)
    open_examples = [
        x for x in all_data 
        if x.get("type", "").upper() == "OPEN" 
        and x.get("split") == "test"
    ][:20]  # Test 20 examples
    
    print(f"Found {len(closed_examples)} closed examples and {len(open_examples)} open examples")
    
    # Test closed model
    if closed_examples:
        closed_acc, closed_results, closed_alpha_imgs, closed_alpha_txts = test_examples(
            "checkpoints/best_closed_improved.pt",
            closed_examples,
            model_type="closed",
            fusion_type="film"
        )
    
    # Test top-k model (with semantic similarity)
    if open_examples:
        topk_acc, topk_results, topk_alpha_imgs, topk_alpha_txts = test_examples(
            "checkpoints/best_topk50_improved.pt",
            open_examples,
            model_type="top-k",
            fusion_type="film",
            use_semantic=True  # Use semantic matching for top-k
        )
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if closed_examples:
        print(f"Closed Model Accuracy: {closed_acc:.1%} ({len(closed_examples)} examples)")
        print(f"  Avg Gate: α_img={sum(closed_alpha_imgs)/len(closed_alpha_imgs):.3f}, α_txt={sum(closed_alpha_txts)/len(closed_alpha_txts):.3f}")
    if open_examples:
        print(f"Top-K Model Accuracy: {topk_acc:.1%} ({len(open_examples)} examples)")
        print(f"  Avg Gate: α_img={sum(topk_alpha_imgs)/len(topk_alpha_imgs):.3f}, α_txt={sum(topk_alpha_txts)/len(topk_alpha_txts):.3f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

