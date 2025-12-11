#!/bin/bash
# Training script for improved open-ended (top-k) model accuracy

echo "=" | head -c 70; echo ""
echo "Training Improved Top-K Model for Open-Ended Questions"
echo "=" | head -c 70; echo ""
echo ""

# Configuration
TOP_K=${1:-100}  # Default to 100, can override: ./train_topk_improved.sh 200
EPOCHS=${2:-20}   # Default to 20 epochs

echo "Configuration:"
echo "  Vocabulary Size: Top-${TOP_K}"
echo "  Epochs: ${EPOCHS}"
echo "  Fusion: Cross-Attention"
echo "  Loss: Focal Loss (gamma=2.5)"
echo "  LR Scheduler: Cosine Annealing"
echo ""

# Training command
python train.py \
  --top_k ${TOP_K} \
  --epochs ${EPOCHS} \
  --batch 8 \
  --lr 5e-4 \
  --lr_scheduler cosine \
  --warmup_epochs 2 \
  --focal_loss \
  --focal_gamma 2.5 \
  --label_smooth 0.1 \
  --fusion_type cross_attn \
  --alpha_entropy 0.03 \
  --save checkpoints/topk${TOP_K}_improved.pt \
  2>&1 | tee train_topk${TOP_K}_improved.log

echo ""
echo "=" | head -c 70; echo ""
echo "Training Complete!"
echo "=" | head -c 70; echo ""
echo ""
echo "Next steps:"
echo "1. Evaluate: python evaluate.py --ckpt checkpoints/best_topk${TOP_K}_improved.pt --top_k ${TOP_K} --fusion_type cross_attn"
echo "2. Test: python test_models.py"
echo "3. Update app.py to use new checkpoint"

