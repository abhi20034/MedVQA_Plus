# app.py - Flask web application for MedVQA+
import os
import re
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
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
from semantic_eval import SemanticMatcher

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device
device = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Global models (loaded on startup)
closed_model = None
topk_model = None
semantic_matcher = None
rag_index = None

# Tokenizer cache (avoid reloading per request)
tokenizer = None

# Question type detection patterns
CLOSED_PATTERNS = [
    r'\b(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\s+',
    r'\b(is\s+there|are\s+there|does\s+it|do\s+they)\s+',
    r'\?$.*\b(yes|no)\b',
]

OPEN_PATTERNS = [
    r'\b(what|where|when|who|which|why|how)\s+',
    r'\b(describe|identify|name|list|show|indicate)\s+',
]


def is_closed_question(question):
    """Detect if question is closed (yes/no) or open-ended."""
    question_lower = question.lower().strip()
    
    # PRIORITY 1: Check for open-ended patterns FIRST (they override closed patterns)
    # Broaden verbs to avoid misrouting descriptive questions.
    expanded_open_patterns = OPEN_PATTERNS + [
        r'\b(explain|compare|summarize|characterize|estimate|measure)\s+',
        r'\b(location|extent|severity)\b'
    ]
    for pattern in expanded_open_patterns:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return False
    
    # PRIORITY 2: Check for explicit yes/no in question text
    # Only check if question explicitly asks about yes/no (e.g., "Is it yes or no?")
    if re.search(r'\b(yes|no)\b', question_lower) and question_lower.endswith('?'):
        # But make sure it's not part of an open-ended question
        if not any(re.search(pattern, question_lower, re.IGNORECASE) for pattern in OPEN_PATTERNS):
            return True
    
    # PRIORITY 3: Check for closed question patterns
    for pattern in CLOSED_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    
    # PRIORITY 4: Default check - if starts with is/are/does/do, likely closed
    if re.match(r'^(is|are|does|do|was|were|has|have|can|could|will|would|should)\s+', question_lower):
        return True
    
    # Otherwise, assume open-ended
    return False


def load_model(ckpt_path, fusion_type="film"):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    answer2id = ckpt["answer2id"]
    id2answer = {v: k for k, v in answer2id.items()}
    num_classes = len(answer2id)
    
    vision = VisionEncoder(proj_dim=768, freeze=True).to(device)
    textenc = TextEncoder(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        proj_dim=768,
        freeze=True
    ).to(device)
    
    if fusion_type == "cross_attn":
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def build_transform():
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def detect_modality(question):
    """Detect imaging modality from question."""
    question_lower = question.lower()
    
    # Check for neuro/brain indicators first (more specific)
    if any(word in question_lower for word in ['brain', 'neuro', 'cerebral', 'cerebellum', 'ventricle', 'lesion', 'infarct', 'hemorrhage', 'mri', 'ct head', 'skull', 'cortex']):
        return 'neuro'
    elif any(word in question_lower for word in ['chest', 'lung', 'pulmonary', 'pleural', 'heart', 'cardiac', 'thorax', 'mediastinum']):
        return 'chest'
    elif any(word in question_lower for word in ['abdomen', 'abdominal', 'liver', 'kidney', 'pancreas', 'gallbladder', 'bowel', 'stomach']):
        return 'abdomen'
    else:
        return 'chest'  # default


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global tokenizer
    cleanup_path = None
    try:
        # Check if image is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        question = request.form.get('question', '').strip()
        modality = request.form.get('modality', 'chest').lower()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
        
        # Save uploaded file with unique suffix to avoid collisions
        filename = secure_filename(file.filename)
        base, ext = os.path.splitext(filename)
        unique_name = f"{base}_{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)
        cleanup_path = filepath
        
        # Detect question type
        is_closed = is_closed_question(question)
        question_type = "closed" if is_closed else "open-ended"
        
        # Auto-detect modality if not provided
        if modality == 'auto':
            modality = detect_modality(question)
        
        # Choose model
        if is_closed:
            model_parts = closed_model
            model_name = "Closed (Yes/No) Model"
        else:
            model_parts = topk_model
            model_name = "Open-Ended (Top-K) Model"
        
        if model_parts is None:
            return jsonify({'error': f'{model_name} not loaded'}), 500
        
        vision, textenc, fusion, head, id2answer, answer2id = model_parts
        
        # Preprocess image
        transform = build_transform()
        img = Image.open(filepath).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Tokenize question (cached tokenizer)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
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
            pred_answer = id2answer.get(pred_id, "<unk>")
            
            alpha_img = float(alpha[0, 0])
            alpha_txt = float(alpha[0, 1])
        
        # For open-ended questions, only use semantic matching if model predicts "yes"/"no"
        # Otherwise, trust the model's direct prediction
        semantic_match_info = None
        if not is_closed and semantic_matcher is not None:
            # Only use semantic matching if the model predicted "yes" or "no" (generic answers)
            if pred_answer.lower() in ['yes', 'no']:
                try:
                    # Get top predictions from the model
                    top_k_probs, top_k_ids = torch.topk(logits[0], k=min(5, len(id2answer)))
                    candidate_answers = [id2answer.get(int(idx.item()), "<unk>") for idx in top_k_ids]
                    
                    # Filter out "yes"/"no" for open-ended questions
                    filtered_candidates = [a for a in candidate_answers if a.lower() not in ['yes', 'no']]
                    if not filtered_candidates:
                        filtered_candidates = candidate_answers  # Fallback if all are yes/no
                    
                    # Filter candidates by modality relevance to avoid cross-modality mismatches
                    modality_keywords = {
                        'chest': ['chest', 'lung', 'pulmonary', 'pleural', 'heart', 'cardiac', 'thorax', 'mediastinum'],
                        'neuro': ['brain', 'cerebral', 'cerebellum', 'ventricle', 'lesion', 'infarct', 'hemorrhage', 'mri', 'neuro', 'skull', 'cortex', 'white matter', 'gray matter'],
                        'abdomen': ['abdomen', 'liver', 'kidney', 'pancreas', 'gallbladder', 'bowel', 'stomach', 'spleen', 'appendix']
                    }
                    
                    # Get modality-specific keywords
                    modality_keywords_list = modality_keywords.get(modality, [])
                    
                    # Score candidates by modality relevance
                    def modality_relevance_score(answer):
                        answer_lower = answer.lower()
                        score = 0
                        for keyword in modality_keywords_list:
                            if keyword in answer_lower:
                                score += 1
                        return score
                    
                    # Sort candidates by modality relevance (higher is better)
                    scored_candidates = [(c, modality_relevance_score(c)) for c in filtered_candidates]
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    # Build query that includes modality context for better matching
                    context_query = f"{question} in {modality} imaging"
                    
                    # Use semantic matching with context-aware query
                    best_match, sim_score = semantic_matcher.best_match(
                        context_query,
                        filtered_candidates, 
                        threshold=0.65
                    )
                    
                    # Additional check: if best match has low modality relevance and there's a better modality-relevant option, prefer it
                    if scored_candidates and scored_candidates[0][1] > 0:  # If there's a modality-relevant candidate
                        best_modality_candidate = scored_candidates[0][0]
                        if best_modality_candidate != best_match:
                            # Check similarity of modality-relevant candidate
                            mod_sim = semantic_matcher.similarity(context_query, best_modality_candidate)
                            # If modality-relevant candidate is close enough, prefer it
                            if mod_sim >= 0.6 and mod_sim >= sim_score - 0.1:  # Within 0.1 of best match
                                best_match = best_modality_candidate
                                sim_score = mod_sim
                    
                    if best_match and sim_score > 0.6:
                        pred_answer = best_match
                        semantic_match_info = {
                            'original_prediction': id2answer.get(pred_id, "<unk>"),
                            'semantic_match': best_match,
                            'similarity_score': round(sim_score, 3),
                            'candidates_checked': len(filtered_candidates),
                            'modality_used': modality
                        }
                except Exception as e:
                    print(f"Semantic matching error: {e}")
            # If model predicted something other than yes/no, use the direct prediction (no semantic matching)
        
        # Get RAG context if available
        rag_context = None
        if rag_index is not None:
            try:
                # Determine if image or text is dominant
                is_image_dominant = alpha_img > alpha_txt
                
                # Build adaptive query based on gate behavior and question type
                if is_closed:
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
                    # Open-ended: describe finding - focus on the predicted answer
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
                    f"{query_focus}"
                )
                
                rag_results = rag_index.search(rag_query, k=3)
                rag_context = [
                    {
                        'text': r['text'][:200] + '...' if len(r['text']) > 200 else r['text'],
                        'source': r['source']
                    }
                    for r in rag_results
                ]
                
                # If gate is balanced, retrieve additional complementary evidence
                if abs(alpha_img - alpha_txt) < 0.15:  # Balanced gate (within 15%)
                    complementary_query = (
                        f"Modality: {modality}. Finding: {pred_answer}. "
                        f"{'Clinical context and diagnostic criteria' if is_image_dominant else 'Imaging appearance and radiologic features'}"
                    )
                    complementary_results = rag_index.search(complementary_query, k=1)
                    for r in complementary_results:
                        rag_context.append({
                            'text': r['text'][:200] + '...' if len(r['text']) > 200 else r['text'],
                            'source': r['source'] + ' (complementary)'
                        })
            except Exception as e:
                print(f"RAG error: {e}")
        
        # Prepare response
        response = {
            'success': True,
            'question': question,
            'question_type': question_type,
            'model_used': model_name,
            'predicted_answer': pred_answer,
            'gate_values': {
                'alpha_img': round(alpha_img, 3),
                'alpha_txt': round(alpha_txt, 3),
                'dominant': 'image' if alpha_img > alpha_txt else 'text'
            },
            'rag_context': rag_context,
            'modality': modality
        }
        
        # Add semantic matching info if available
        if semantic_match_info:
            response['semantic_matching'] = semantic_match_info
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except Exception:
                pass


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'closed_model_loaded': closed_model is not None,
        'topk_model_loaded': topk_model is not None,
        'rag_loaded': rag_index is not None,
        'device': str(device)
    }
    return jsonify(status)


if __name__ == '__main__':
    print("=" * 70)
    print("Loading MedVQA+ Models...")
    print("=" * 70)
    
    # Preload tokenizer once to avoid per-request cost
    try:
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"⚠ Warning: Could not preload tokenizer: {e}")
    
    # Load closed model
    closed_ckpt = "checkpoints/best_closed_improved.pt"
    if os.path.exists(closed_ckpt):
        print(f"Loading closed model from {closed_ckpt}...")
        closed_model = load_model(closed_ckpt, fusion_type="film")
        print("✓ Closed model loaded")
    else:
        print(f"⚠ Warning: {closed_ckpt} not found")
    
    # Load top-k model
    topk_ckpt = "checkpoints/best_topk50_improved.pt"
    if os.path.exists(topk_ckpt):
        print(f"Loading top-k model from {topk_ckpt}...")
        topk_model = load_model(topk_ckpt, fusion_type="film")
        print("✓ Top-k model loaded")
    else:
        print(f"⚠ Warning: {topk_ckpt} not found")
    
    # Load semantic matcher (for future use)
    print("Loading semantic matcher...")
    try:
        semantic_matcher = SemanticMatcher(device=device)
        print("✓ Semantic matcher loaded")
    except Exception as e:
        print(f"⚠ Warning: Could not load semantic matcher: {e}")
    
    # Load RAG index
    print("Loading RAG index...")
    try:
        rag_index = RAGIndex(
            corpus_dir="retrieval/medical_corpus",
            model_name="emilyalsentzer/Bio_ClinicalBERT",
            device=device,
        )
        rag_index.build_from_corpus(use_cache=True)
        print("✓ RAG index loaded")
    except Exception as e:
        print(f"⚠ Warning: Could not load RAG index: {e}")
    
    print("=" * 70)
    print("Starting Flask server...")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
