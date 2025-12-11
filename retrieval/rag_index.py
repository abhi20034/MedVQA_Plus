# retrieval/rag_index.py

import os
import pickle
import hashlib
from typing import List, Dict, Optional

import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel


class RAGIndex:
    """
    Simple RAG index:
      - loads all .txt files from medical_corpus/
      - splits into paragraph chunks
      - encodes each chunk with a clinical BERT model
      - builds a FAISS index for similarity search

    You can restrict which .txt files are used via `allowed_sources`.
    """

    def __init__(
        self,
        corpus_dir: str = "retrieval/medical_corpus",
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: Optional[str] = None,
        max_len: int = 96,
        allowed_sources: Optional[List[str]] = None,
    ):
        self.corpus_dir = corpus_dir
        self.model_name = model_name
        self.max_len = max_len
        self.allowed_sources = allowed_sources  # list of filenames (e.g. ["chest_imaging.txt"])

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.lm = AutoModel.from_pretrained(model_name).to(self.device)
        self.lm.eval()

        self.texts: List[str] = []
        self.sources: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        
        # Caching support
        self.cache_dir = os.path.join(corpus_dir, "..", ".rag_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # --------- helper: mean pool ----------
    def _mean_pool(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-6)
        return summed / denom

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        all_embs = []
        bs = 16
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i + bs]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                ).to(self.device)
                out = self.lm(**enc)
                emb = self._mean_pool(out, enc["attention_mask"])  # [B, H]
                all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0).numpy()  # [N, H]

    def _get_cache_path(self):
        """Generate cache file path based on configuration."""
        cache_key_parts = [
            self.corpus_dir,
            self.model_name,
            str(sorted(self.allowed_sources) if self.allowed_sources else "all"),
            str(self.max_len),
        ]
        cache_key = "_".join(cache_key_parts)
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_file = os.path.join(self.cache_dir, f"rag_index_{cache_hash}.pkl")
        return cache_file
    
    def _load_from_cache(self) -> bool:
        """Load index from cache if available. Returns True if loaded successfully."""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.texts = data["texts"]
                self.sources = data["sources"]
                self.embeddings = data["embeddings"]
                
                dim = self.embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(self.embeddings.astype(np.float32))
                
            print(f"[RAG] Loaded index from cache: {len(self.texts)} chunks")
            return True
        except Exception as e:
            print(f"[RAG] Cache load failed: {e}, rebuilding...")
            return False
    
    def _save_to_cache(self):
        """Save index to cache."""
        cache_path = self._get_cache_path()
        try:
            data = {
                "texts": self.texts,
                "sources": self.sources,
                "embeddings": self.embeddings,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[RAG] Saved index to cache: {cache_path}")
        except Exception as e:
            print(f"[RAG] Cache save failed: {e}")
    
    # --------- public API ----------
    def build_from_corpus(self, use_cache: bool = True):
        """
        Loads all .txt files in corpus_dir, splits on blank lines into chunks,
        encodes, and builds a FAISS index.

        If `allowed_sources` is not None, only those filenames are used.
        
        Args:
            use_cache: If True, try to load from cache first, save after building.
        """
        # Try loading from cache
        if use_cache and self._load_from_cache():
            return
        texts: List[str] = []
        sources: List[str] = []

        for fname in os.listdir(self.corpus_dir):
            if not fname.endswith(".txt"):
                continue
            if self.allowed_sources is not None and fname not in self.allowed_sources:
                continue  # skip files not in the allowed list

            fpath = os.path.join(self.corpus_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                raw = f.read()

            # Split into paragraphs separated by blank lines
            for chunk in raw.split("\n\n"):
                c = chunk.strip()
                if len(c) < 32:  # skip very tiny lines
                    continue
                texts.append(c)
                sources.append(fname)

        if not texts:
            raise RuntimeError(f"[RAG] No usable text chunks found in {self.corpus_dir} with allowed={self.allowed_sources}")

        print(f"[RAG] Loaded {len(texts)} chunks from {self.corpus_dir}")
        self.texts = texts
        self.sources = sources

        print("[RAG] Encoding chunks with clinical BERT...")
        embs = self._encode_texts(texts)  # [N, H]
        self.embeddings = embs
        dim = embs.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embs.astype(np.float32))
        self.index = index
        print(f"[RAG] Built FAISS index with {self.index.ntotal} vectors (dim={dim})")
        
        # Save to cache
        if use_cache:
            self._save_to_cache()

    @torch.no_grad()
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Returns list of dicts: {text, source, score}, sorted by relevance.
        """
        if self.index is None:
            raise RuntimeError("RAG index not built. Call build_from_corpus() first.")

        enc = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)

        out = self.lm(**enc)
        q_emb = self._mean_pool(out, enc["attention_mask"]).cpu().numpy().astype(np.float32)  # [1, H]

        D, I = self.index.search(q_emb, k)
        idxs = I[0]
        dists = D[0]

        results: List[Dict] = []
        for idx, dist in zip(idxs, dists):
            idx = int(idx)
            results.append({
                "text": self.texts[idx],
                "source": self.sources[idx],
                "score": float(-dist),  # negative L2 distance as similarity
            })
        return results