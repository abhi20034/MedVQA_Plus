# semantic_eval.py
"""
Semantic similarity evaluation for VQA models.
Uses ClinicalBERT embeddings to compute similarity between predicted and true answers.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class SemanticMatcher:
    """Compute semantic similarity between answers using ClinicalBERT."""
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def _mean_pool(self, outputs, attention_mask):
        """Mean pooling over sequence length."""
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-6)
        return summed / denom
    
    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,  # Shorter for answers
            return_tensors="pt",
        ).to(self.device)
        
        out = self.model(**enc)
        emb = self._mean_pool(out, enc["attention_mask"])
        return emb.cpu().numpy()
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embs = self.encode([text1, text2])
        emb1, emb2 = embs[0], embs[1]
        
        # Cosine similarity
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        sim = dot / (norm1 * norm2 + 1e-8)
        return float(sim)
    
    def best_match(self, pred: str, candidates: List[str], threshold: float = 0.7) -> Tuple[str, float]:
        """
        Find best matching candidate for prediction.
        Returns: (best_match, similarity_score)
        """
        if not candidates:
            return pred, 0.0
        
        # Encode all at once for efficiency
        all_texts = [pred] + candidates
        embs = self.encode(all_texts)
        pred_emb = embs[0]
        cand_embs = embs[1:]
        
        # Compute similarities
        similarities = []
        for cand_emb in cand_embs:
            dot = np.dot(pred_emb, cand_emb)
            norm1 = np.linalg.norm(pred_emb)
            norm2 = np.linalg.norm(cand_emb)
            sim = dot / (norm1 * norm2 + 1e-8)
            similarities.append(float(sim))
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= threshold:
            return candidates[best_idx], best_sim
        else:
            return pred, best_sim  # Return original if no good match

