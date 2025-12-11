# MedVQA+ - Medical Visual Question Answering System

A comprehensive medical VQA system with automatic model selection, RAG integration, and gate-based multi-modal fusion.

## 📋 Table of Contents

1. [Installation](#installation)
2. [Dataset Information](#dataset-information)
3. [Training Models](#training-models)
4. [Evaluating Models](#evaluating-models)
5. [Testing Models](#testing-models)
6. [Gate Analysis](#gate-analysis)
7. [RAG-Enhanced Prediction](#rag-enhanced-prediction)
8. [Web Application](#web-application)
9. [Model Architecture](#model-architecture)
10. [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA/MPS support (optional but recommended)

### Setup

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install torch torchvision transformers pillow numpy faiss-cpu scikit-learn

# For web application
pip install Flask Werkzeug
```

---

## 📊 Dataset Information

**Dataset**: VQA-RAD  
**Total Samples**: 2,248  
**Split**: 80% Train / 10% Validation / 10% Test

- **Train**: 1,802 samples (57.8% closed, 42.2% open)
- **Validation**: 223 samples (57.8% closed, 42.2% open)
- **Test**: 223 samples (57.8% closed, 42.2% open)

The dataset is pre-split in `data/VQA_RAD/qa_pairs.json` with stratified sampling.

---

## 🎓 Training Models

### 1. Training Closed-Only Model (Yes/No Questions)

**Purpose**: Train a binary classifier for closed-ended (yes/no) questions.

**Command**:
```bash
python train.py \
  --closed_only \
  --epochs 12 \
  --batch 8 \
  --lr 5e-4 \
  --lr_scheduler cosine \
  --label_smooth 0.05 \
  --alpha_entropy 0.03 \
  --save checkpoints/best_closed_improved.pt \
  2>&1 | tee train_closed.log
```

**Explanation**:
- `--closed_only`: Filters dataset to only yes/no questions (2 classes: "yes", "no")
- `--epochs 12`: Train for 12 epochs
- `--batch 8`: Batch size of 8 (adjust based on GPU memory)
- `--lr 5e-4`: Learning rate of 0.0005
- `--lr_scheduler cosine`: Use cosine annealing LR scheduler for better convergence
- `--label_smooth 0.05`: Apply label smoothing (5%) to prevent overfitting
- `--alpha_entropy 0.03`: Gate entropy regularization weight (encourages balanced gate usage)
- `--save`: Path to save checkpoints (best and final models saved automatically)
- `2>&1 | tee train_closed.log`: Save training logs to file

**Expected Output**: 
- Best model saved to `checkpoints/best_closed_improved.pt`
- Final model saved to `checkpoints/closed_improved.pt`
- Training log in `train_closed.log`

---

### 2. Training Top-K Model (Open-Ended Questions)

**Purpose**: Train a multi-class classifier for open-ended questions using top-K most frequent answers.

**Command**:
```bash
python train.py \
  --top_k 50 \
  --epochs 12 \
  --batch 8 \
  --lr 5e-4 \
  --lr_scheduler cosine \
  --label_smooth 0.05 \
  --alpha_entropy 0.03 \
  --save checkpoints/best_topk50_improved.pt \
  2>&1 | tee train_topk50.log
```

**Explanation**:
- `--top_k 50`: Use only top-50 most frequent answers (reduces vocabulary from 500+ to 50)
- `--top_k 0`: Use all answers (full vocabulary, slower training)
- Other parameters same as closed model

**Expected Output**:
- Best model: `checkpoints/best_topk50_improved.pt`
- Final model: `checkpoints/topk50_improved.pt`
- Training log: `train_topk50.log`

---

### 3. Training with Advanced Features

**Purpose**: Use advanced training techniques for better accuracy.

**Command**:
```bash
python train.py \
  --closed_only \
  --epochs 15 \
  --batch 8 \
  --lr 5e-4 \
  --lr_scheduler cosine \
  --focal_loss \
  --focal_gamma 2.0 \
  --label_smooth 0.1 \
  --fusion_type cross_attn \
  --alpha_entropy 0.03 \
  --save checkpoints/advanced_closed.pt \
  2>&1 | tee train_advanced.log
```

**Explanation**:
- `--focal_loss`: Use Focal Loss instead of Cross-Entropy (better for class imbalance)
- `--focal_gamma 2.0`: Focal loss focusing parameter (higher = more focus on hard examples)
- `--label_smooth 0.1`: Higher label smoothing (10%) for stronger regularization
- `--fusion_type cross_attn`: Use cross-attention fusion instead of FiLM (more sophisticated)

**When to Use**:
- Use `--focal_loss` when you have severe class imbalance
- Use `--fusion_type cross_attn` for complex multi-modal relationships (slower but potentially better)
- Use higher `--label_smooth` if model is overfitting

---

### 4. Training Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | `data` | Root directory containing VQA_RAD dataset |
| `--epochs` | `12` | Number of training epochs |
| `--batch` | `8` | Batch size (adjust based on GPU memory) |
| `--lr` | `5e-4` | Learning rate |
| `--txt_model` | `emilyalsentzer/Bio_ClinicalBERT` | Text encoder model |
| `--freeze` | `False` | Freeze encoder weights (faster but less flexible) |
| `--closed_only` | `False` | Use only yes/no questions |
| `--top_k` | `0` | Use top-K answers (0 = all) |
| `--lr_scheduler` | `cosine` | LR scheduler: `none`, `cosine`, `step` |
| `--label_smooth` | `0.0` | Label smoothing coefficient (0-1) |
| `--focal_loss` | `False` | Use Focal Loss |
| `--focal_gamma` | `2.0` | Focal loss gamma parameter |
| `--fusion_type` | `film` | Fusion type: `film` or `cross_attn` |
| `--alpha_tau` | `2.0` | Gate softmax temperature (>1 = softer) |
| `--alpha_entropy` | `0.02` | Gate entropy regularization weight |
| `--save` | `checkpoints/baseline.pt` | Checkpoint save path |

---

## 📈 Evaluating Models

### 1. Evaluate Closed Model

**Purpose**: Calculate test accuracy for closed (yes/no) model.

**Command**:
```bash
python evaluate.py \
  --ckpt checkpoints/best_closed_improved.pt \
  --closed_only \
  --batch 8 \
  --fusion_type film
```

**Explanation**:
- `--ckpt`: Path to checkpoint file
- `--closed_only`: Must match training setting
- `--fusion_type`: Must match training setting (`film` or `cross_attn`)
- `--batch`: Batch size for evaluation

**Output**: Test accuracy printed to console.

---

### 2. Evaluate Top-K Model

**Purpose**: Calculate test accuracy for top-K model.

**Command**:
```bash
python evaluate.py \
  --ckpt checkpoints/best_topk50_improved.pt \
  --top_k 50 \
  --batch 8 \
  --fusion_type film
```

**Explanation**:
- `--top_k 50`: Must match training setting
- Other parameters same as above

---

## 🧪 Testing Models

### 1. Test Both Models with Multiple Examples

**Purpose**: Test both closed and top-k models on multiple examples with semantic similarity matching.

**Command**:
```bash
python test_models.py
```

**Explanation**:
- Tests 20 closed examples and 20 open-ended examples
- Uses semantic similarity matching for top-k model (threshold=0.7)
- Shows accuracy, gate values, and detailed results
- Prints results to console

**Output**:
- Closed model accuracy
- Top-k model accuracy (with semantic matching)
- Average gate values
- Per-example results

---

### 2. Test Individual Examples with RAG

**Purpose**: Test model on a single image/question with RAG-enhanced context.

**Command**:
```bash
python predict_with_rag.py \
  --image data/VQA_RAD/images/synpic100132.jpg \
  --question "Is there a pleural effusion?" \
  --ckpt checkpoints/best_closed_improved.pt \
  --modality chest \
  --k 3 \
  --fusion_type film
```

**Explanation**:
- `--image`: Path to medical image
- `--question`: Question to answer
- `--ckpt`: Model checkpoint path
- `--modality`: Imaging modality (`chest`, `neuro`, `abdomen`, `other`)
- `--k 3`: Number of RAG evidence chunks to retrieve
- `--fusion_type`: Must match training setting

**Output**:
- Predicted answer
- Gate values (alpha_img, alpha_txt)
- Gate behavior (image-dominant vs text-dominant)
- RAG-retrieved clinical context
- Clinical interpretation guidance

---

### 3. Test Open-Ended Questions

**Purpose**: Test top-k model on open-ended questions.

**Command**:
```bash
python predict_with_rag.py \
  --image data/VQA_RAD/images/synpic12210.jpg \
  --question "What is the location of the lesion?" \
  --ckpt checkpoints/best_topk50_improved.pt \
  --modality neuro \
  --k 3 \
  --fusion_type film
```

**Explanation**: Same as above but using top-k model for open-ended questions.

---

## 🔍 Gate Analysis

### 1. Analyze Gate Behavior

**Purpose**: Analyze how the gate (alpha values) behaves across the dataset.

**Command**:
```bash
python analyze_gate.py \
  --ckpt checkpoints/best_closed_improved.pt \
  --data_root data \
  --closed_only \
  --split test \
  --max_samples 500 \
  --fusion_type film
```

**Explanation**:
- `--ckpt`: Model checkpoint path
- `--data_root`: Dataset root directory
- `--closed_only`: Analyze closed model (use `--top_k 50` for top-k model)
- `--split`: Dataset split to analyze (`train`, `val`, `test`)
- `--max_samples`: Maximum samples to analyze (0 = all)
- `--fusion_type`: Must match training setting

**Output**:
- Gate statistics (mean, std, min, max)
- Entropy distribution
- Gate balance analysis
- Visualization saved to `gate_analysis_test.png`

---

### 2. Gate Analysis for Top-K Model

**Command**:
```bash
python analyze_gate.py \
  --ckpt checkpoints/best_topk50_improved.pt \
  --data_root data \
  --top_k 50 \
  --split test \
  --max_samples 500 \
  --fusion_type film
```

**Explanation**: Same as above but for top-k model.

---

## 🔬 RAG-Enhanced Prediction

### 1. RAG Index Building

**Purpose**: Build RAG index from medical corpus (automatic on first use).

The RAG index is automatically built and cached on first use. It uses:
- Medical corpus in `retrieval/medical_corpus/`
- ClinicalBERT for embeddings
- FAISS for similarity search

**Manual rebuild** (if needed):
```python
from retrieval.rag_index import RAGIndex
rag = RAGIndex(corpus_dir="retrieval/medical_corpus")
rag.build_from_corpus(use_cache=False)  # Force rebuild
```

---

### 2. RAG Query Types

The RAG system adapts queries based on:
- **Question type** (closed vs open-ended)
- **Gate behavior** (image-dominant vs text-dominant)
- **Modality** (chest, neuro, abdomen)

**Example RAG Queries**:
- Image-dominant closed: "Radiologic imaging findings of yes/no [question]..."
- Text-dominant closed: "Clinical interpretation and diagnostic criteria..."
- Open-ended: "Radiologic appearance and imaging characteristics..."

---

## 🌐 Web Application

### 1. Start Web Server

**Purpose**: Launch interactive web interface for model inference.

**Command**:
```bash
python app.py
```

**Explanation**:
- Starts Flask server on port 5001 (changed from 5000 to avoid macOS AirPlay conflict)
- Loads both closed and top-k models automatically
- Loads RAG index and semantic matcher
- Serves web interface at `http://localhost:5001`

**Access**: Open browser to `http://localhost:5001`

---

### 2. Web Application Features

- **Automatic Model Selection**: Detects question type and selects appropriate model
- **Image Upload**: Drag-and-drop or click to upload
- **Real-time Results**: Shows predictions, gate values, and RAG context
- **Modality Selection**: Auto-detect or manual selection

---

### 3. Web API Endpoints

**Health Check**:
```bash
curl http://localhost:5001/health
```

**Prediction** (POST):
```bash
curl -X POST http://localhost:5001/predict \
  -F "image=@path/to/image.jpg" \
  -F "question=Is there a pleural effusion?" \
  -F "modality=chest"
```

---

## 🏗️ Model Architecture

### Components

1. **VisionEncoder**: CLIP-based image encoder
   - Input: Medical images (224x224)
   - Output: Image embeddings (768-dim)

2. **TextEncoder**: ClinicalBERT-based text encoder
   - Input: Questions (tokenized)
   - Output: Text embeddings (768-dim)

3. **Fusion Module**: 
   - **FiLMFusion** (default): Feature-wise modulation
   - **CrossAttentionFusion**: Bidirectional cross-attention

4. **DecisionScaler**: Adaptive gate mechanism
   - Computes alpha values (image vs text weights)
   - Fuses features based on gate values
   - Classifies to answer vocabulary

5. **RAG Index**: Retrieval-augmented generation
   - Medical corpus embeddings
   - FAISS similarity search
   - Clinical context retrieval

---

## 📝 Training Tips

### 1. Hyperparameter Tuning

**Learning Rate**:
- Start with `5e-4`
- Increase to `1e-3` if training is slow
- Decrease to `1e-4` if loss is unstable

**Batch Size**:
- GPU with 8GB: `batch=8`
- GPU with 16GB: `batch=16`
- GPU with 24GB+: `batch=32`

**Epochs**:
- Closed model: 12-15 epochs usually sufficient
- Top-k model: 15-20 epochs for better convergence

### 2. Regularization

**Label Smoothing**:
- Start with `0.05` (5%)
- Increase to `0.1` if overfitting
- Use `0.0` for no smoothing

**Gate Entropy**:
- Default: `0.02`
- Increase to `0.03-0.05` to encourage balanced gate
- Decrease to `0.01` if gate is too uniform

### 3. Loss Functions

**Cross-Entropy** (default):
- Good for balanced datasets
- Fast and stable

**Focal Loss**:
- Better for imbalanced datasets
- Use `--focal_gamma 2.0` for moderate focus
- Use `--focal_gamma 2.5-3.0` for strong focus on hard examples

---

## 🐛 Troubleshooting

### 1. Out of Memory Errors

**Solution**:
- Reduce batch size: `--batch 4` or `--batch 2`
- Use gradient accumulation (not implemented, but can add)
- Use smaller model or freeze encoders: `--freeze`

### 2. Model Not Loading

**Check**:
- Checkpoint path is correct
- `--fusion_type` matches training setting
- `--closed_only` or `--top_k` matches training setting

### 3. Low Accuracy

**Try**:
- Train for more epochs: `--epochs 20`
- Use cosine LR scheduler: `--lr_scheduler cosine`
- Add label smoothing: `--label_smooth 0.1`
- Use focal loss: `--focal_loss --focal_gamma 2.0`
- Increase gate entropy: `--alpha_entropy 0.05`

### 4. Gate Bias Issues

**If gate is too biased** (e.g., always image-dominant):
- Increase entropy regularization: `--alpha_entropy 0.05`
- Check gate initialization in `models/decision_scale.py`
- Verify gate normalization is working

### 5. Port Already in Use

**Solution**:
- Change port in `app.py`: `app.run(port=5002)`
- Kill existing process: `kill $(lsof -ti:5001)`
- On macOS: Disable AirPlay Receiver (uses port 5000)

---

## 📊 Expected Results

### Closed Model
- **Accuracy**: 70-80% on test set
- **Gate Balance**: α_img ≈ 0.5, α_txt ≈ 0.5

### Top-K Model (50 classes)
- **Accuracy**: 40-50% (exact match)
- **Accuracy**: 65-75% (semantic similarity, threshold=0.7)
- **Gate Balance**: α_img ≈ 0.5-0.6, α_txt ≈ 0.4-0.5

---

## 🔗 File Structure

```
MedVQA_Plus/
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
├── test_models.py        # Comprehensive testing
├── analyze_gate.py        # Gate behavior analysis
├── predict_with_rag.py   # RAG-enhanced prediction
├── app.py                # Web application
├── semantic_eval.py      # Semantic similarity matching
├── dataset_loader.py     # Data loading and preprocessing
├── models/
│   ├── vision_encoder.py
│   ├── text_encoder.py
│   ├── fusion_model.py
│   ├── cross_attention_fusion.py
│   └── decision_scale.py
├── retrieval/
│   ├── rag_index.py
│   └── medical_corpus/
├── checkpoints/          # Saved models
├── templates/            # Web templates
└── static/               # Web static files
```

---

## 📚 Additional Resources

- **Dataset Split Info**: See `DATASET_SPLIT_INFO.md`
- **Training Results**: See `TRAINING_RESULTS.md`
- **Gate Analysis**: See `GATE_RAG_INTEGRATION.md`
- **Fusion Explanation**: See `FUSION_EXPLANATION.md`
- **Web App Guide**: See `README_WEB.md`

---

## ✅ Quick Start Checklist

1. ✅ Install dependencies
2. ✅ Download/verify VQA-RAD dataset in `data/VQA_RAD/`
3. ✅ Train closed model: `python train.py --closed_only ...`
4. ✅ Train top-k model: `python train.py --top_k 50 ...`
5. ✅ Evaluate models: `python evaluate.py ...`
6. ✅ Test models: `python test_models.py`
7. ✅ Analyze gate: `python analyze_gate.py ...`
8. ✅ Run web app: `python app.py`

---

## 📄 License

Same as main MedVQA+ project.

---

## 🙏 Acknowledgments

- VQA-RAD dataset
- CLIP for vision encoding
- ClinicalBERT for text encoding
- FAISS for similarity search

---

**Last Updated**: 2024  
**Version**: 1.0
