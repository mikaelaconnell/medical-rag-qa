"""
Retrieval module using BiomedBERT bi-encoder and FAISS vector index.
Encodes PubMed abstracts and retrieves top-K relevant passages for a query.
"""

import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional


class BiomedRetriever:
    """Bi-encoder retriever using BiomedBERT embeddings and FAISS."""

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.index = None
        self.documents = []
        self.metadata = []

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into dense embeddings using mean pooling.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show a progress bar

        Returns:
            numpy array of shape (len(texts), hidden_dim)
        """
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding")

        for i in iterator:
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean pooling over non-padding tokens
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1)
            embeddings = embeddings.cpu().numpy()

            # L2 normalize for cosine similarity via inner product
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype("float32")

    def build_index(self, documents: List[str], metadata: Optional[List[dict]] = None):
        """
        Build FAISS index from document texts.

        Args:
            documents: List of document texts (abstracts)
            metadata: Optional list of metadata dicts per document
        """
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]

        print(f"Encoding {len(documents)} documents...")
        embeddings = self.encode(documents)

        # Use inner product (equivalent to cosine sim since vectors are normalized)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Retrieve top-K documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of (document_text, score, metadata) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self.encode([query], show_progress=False)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score), self.metadata[idx]))
        return results

    def save_index(self, path: str):
        """Save FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, path)

    def load_index(self, path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
