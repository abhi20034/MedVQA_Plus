# predict_with_rag.py
import os
import argparse
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
from retrieval.rag_index import RAGIndex

TOKENIZER_CACHE = {}


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


def load_model(ckpt_path: str, txt_model: str = "emilyalsentzer/Bio_ClinicalBERT", fusion_type: str = "film"):
    device = get_device()
    print(">>> using device:", device)

    ckpt = torch.load(ckpt_path, map_location=device)

    answer2id = ckpt["answer2id"]
    id2answer = ckpt.get("id2answer", {v: k for k, v in answer2id.items()})
    num_classes = len(answer2id)

    # encoders can be frozen or not – for inference it doesn't matter
    vision = VisionEncoder(proj_dim=768, freeze=True).to(device)
    textenc = TextEncoder(model_name=txt_model, proj_dim=768, freeze=True).to(device)
    
    # Choose fusion type
    if fusion_type == "cross_attn":
        fusion = CrossAttentionFusion(dim=768, num_heads=8, dropout=0.1).to(device)
    else:
        fusion = FiLMFusion(dim=768, hidden=512).to(device)
    
    head   = DecisionScaler(dim=768, num_classes=num_classes).to(device)

    # Load with strict=False for backward compatibility
    vision.load_state_dict(ckpt["vision"], strict=False)
    textenc.load_state_dict(ckpt["textenc"], strict=False)
    fusion.load_state_dict(ckpt["fusion"], strict=False)
    head.load_state_dict(ckpt["head"], strict=False)

    vision.eval(); textenc.eval(); fusion.eval(); head.eval()

    return vision, textenc, fusion, head, id2answer, device


@torch.no_grad()
def answer_with_rag(
    image_path: str,
    question: str,
    ckpt_path: str,
    txt_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    k_evidence: int = 3,
    modality: str = "chest",
    fusion_type: str = "film",
):
    # ---- load model ----
    vision, textenc, fusion, head, id2answer, device = load_model(ckpt_path, txt_model, fusion_type)
    transform = build_transform()

    # ---- image ----
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # ---- tokenize question ----
    tokenizer = TOKENIZER_CACHE.get(txt_model)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(txt_model, use_fast=True)
        TOKENIZER_CACHE[txt_model] = tokenizer
    enc = tokenizer(
        [question],
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="pt",
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # ---- forward pass ----
    e_img = vision(img)
    e_txt = textenc(input_ids, attention_mask)
    z_img = fusion(e_img, e_txt)
    logits, alpha, _ = head(z_img, e_txt)

    pred_id = int(logits.argmax(dim=-1)[0].item())
    pred_answer = id2answer.get(pred_id, "<unk>")

    # ---- gate (α) integration ----
    alpha_img = float(alpha[0, 0])
    alpha_txt = float(alpha[0, 1])
    alpha_vals = [round(alpha_img, 3), round(alpha_txt, 3)]

    print("\n============================")
    print(f"QUESTION: {question}")
    print(f"PREDICTED ANSWER: {pred_answer}")
    print(f"alpha (img, txt): {alpha_vals}")
    if alpha_img > alpha_txt:
        print("Model relied more on IMAGE features.")
    elif alpha_txt > alpha_img:
        print("Model relied more on TEXT features.")
    else:
        print("Model relied roughly equally on image and text.")
    print("============================\n")

    # ---- modality → allowed corpus files ----
    m = modality.lower()
    if m == "chest":
        allowed = ["chest_imaging.txt", "general_radiology_terms.txt"]
    elif m in {"neuro", "brain"}:
        allowed = ["neuro_radiology.txt", "mri_sequences.txt", "general_radiology_terms.txt"]
    elif m in {"abdomen", "abdominal"}:
        allowed = ["abdomen_ct.txt", "general_radiology_terms.txt"]
    else:
        allowed = None  # fallback: all files

    # ---- build/cache RAG index ----
    # RAG index caching is automatically handled by RAGIndex class
    rag = RAGIndex(
        corpus_dir="retrieval/medical_corpus",
        model_name=txt_model,
        device=str(device),
        allowed_sources=allowed,
    )
    rag.build_from_corpus(use_cache=True)  # Uses cache if available, builds if not

    # ---- RAG query that uses α + prediction (gate-aware) ----
    # Determine dominant modality for adaptive query construction
    is_image_dominant = alpha_img > alpha_txt
    alpha_ratio = alpha_img / (alpha_txt + 1e-6)
    
    # Adapt query based on gate behavior and question type
    if pred_answer.lower() in {"yes", "no"}:
        # Closed-ended question: focus on finding confirmation/negation
        if is_image_dominant:
            query_focus = (
                f"Radiologic imaging findings of {pred_answer.lower()} {question.lower()}. "
                f"What are the typical visual characteristics and imaging appearance on {modality}?"
            )
        else:
            query_focus = (
                f"Clinical interpretation and diagnostic criteria for {pred_answer.lower()} "
                f"regarding {question.lower()} in {modality} imaging. "
                f"What are the diagnostic features and clinical significance?"
            )
    else:
        # Open-ended: describe finding
        if is_image_dominant:
            query_focus = (
                f"Radiologic appearance and imaging characteristics of {pred_answer} on {modality}. "
                f"Describe the visual findings, signal intensity, contrast enhancement, and typical location."
            )
        else:
            query_focus = (
                f"Clinical and diagnostic features of {pred_answer} in {modality} imaging. "
                f"What is the typical presentation, differential diagnosis, and clinical significance?"
            )
    
    # Build comprehensive query with gate context
    rag_query = (
        f"Modality: {modality}. "
        f"Clinical question: {question}. "
        f"Predicted answer: {pred_answer}. "
        f"{query_focus} "
        f"[Model confidence: {'Image-heavy' if is_image_dominant else 'Text-heavy'} "
        f"(Image: {alpha_img:.1%}, Text: {alpha_txt:.1%})]"
    )

    results = rag.search(rag_query, k=k_evidence)
    
    # Optionally retrieve additional context if alpha is balanced (multi-modal evidence)
    if abs(alpha_img - alpha_txt) < 0.15:  # Balanced gate (within 15%)
        print(f"[RAG] Balanced gate detected - retrieving complementary evidence...")
        # Could add a second query focused on the other modality
        complementary_query = (
            f"Modality: {modality}. Finding: {pred_answer}. "
            f"{'Clinical context and diagnostic criteria' if is_image_dominant else 'Imaging appearance and radiologic features'}"
        )
        complementary_results = rag.search(complementary_query, k=1)  # Get 1 additional piece
        if complementary_results:
            results.append(complementary_results[0])  # Add to results

    # Filter and format results
    print(f"\n{'='*60}")
    print(f"RAG Retrieval Summary:")
    print(f"{'='*60}")
    print(f"Modality: {modality}")
    print(f"Allowed sources: {allowed if allowed else 'all'}")
    print(f"Gate behavior: {'Image-dominant' if is_image_dominant else 'Text-dominant'} "
          f"(α_img={alpha_img:.2f}, α_txt={alpha_txt:.2f})")
    print(f"Query focus: {'Visual/imaging features' if is_image_dominant else 'Clinical/diagnostic context'}")
    print(f"{'='*60}\n")
    
    print(f"Retrieved evidence ({len(results)} chunks):")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Source: {r['source']}")
        print(f"    Relevance score: {r['score']:.3f}")
        print(f"    Text: {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}")
    
    print(f"\n{'='*60}")
    print("Clinical Interpretation Guidance:")
    print(f"{'='*60}")
    if is_image_dominant:
        print("✓ Model primarily relied on imaging features.")
        print("✓ Focus on visual evidence in retrieved texts.")
        print("✓ Correlate imaging appearance with clinical context.")
    else:
        print("✓ Model primarily relied on question/clinical context.")
        print("✓ Emphasize diagnostic criteria and clinical significance.")
        print("✓ Consider imaging appearance as supporting evidence.")
    print(f"{'='*60}\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="Path to image file")
    ap.add_argument("--question", type=str, required=True, help="Clinical question")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    ap.add_argument("--txt_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--k", type=int, default=3, help="Number of RAG evidence chunks")
    ap.add_argument("--modality", type=str, default="chest",
                    help="Imaging modality: chest / neuro / abdomen / other")
    ap.add_argument("--fusion_type", type=str, default="film", choices=["film", "cross_attn"],
                    help="Fusion type used in checkpoint")
    return ap.parse_args()


if __name__ == "__main__":
    import os as _os
    _os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()
    answer_with_rag(
        image_path=args.image,
        question=args.question,
        ckpt_path=args.ckpt,
        txt_model=args.txt_model,
        k_evidence=args.k,
        modality=args.modality,
        fusion_type=args.fusion_type,
    )
