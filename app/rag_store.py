# app/rag_store.py
import os
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import chromadb
from chromadb.config import Settings

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False

from sentence_transformers import SentenceTransformer


@dataclass
class RagHit:
    text: str
    source: str
    chunk_id: str
    distance: float


class RagStore:
    """
    Local RAG store using ChromaDB (persistent).
    - add_documents(files)
    - query(text) -> top chunks
    - clear()
    - stats()
    """
    def __init__(
        self,
        persist_dir: str = "rag_db",
        collection_name: str = "support_kb",
        embed_model: str = "all-MiniLM-L6-v2"
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.embedder = SentenceTransformer(embed_model)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # ---------- ingestion ----------
    def _read_file_bytes(self, filename: str, b: bytes) -> str:
        lower = filename.lower()

        if lower.endswith(".pdf"):
            if not _HAS_PYPDF:
                return "[PDF uploaded but pypdf not installed. Install pypdf to parse PDFs.]"

            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(b)
                tmp_path = tmp.name
            try:
                reader = PdfReader(tmp_path)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text() or "")
                return "\n".join(text).strip()
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        try:
            return b.decode("utf-8")
        except Exception:
            return b.decode("latin-1", errors="ignore")

    def _chunk_text(self, text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        buf = ""

        for p in paras:
            if len(buf) + len(p) + 1 <= max_chars:
                buf = (buf + "\n" + p).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = p

        if buf:
            chunks.append(buf)

        if overlap > 0 and len(chunks) > 1:
            out = []
            for i, ch in enumerate(chunks):
                if i == 0:
                    out.append(ch)
                else:
                    prev = chunks[i - 1]
                    tail = prev[-overlap:]
                    out.append((tail + "\n" + ch).strip())
            chunks = out

        return chunks

    def add_documents(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = []
        docs = []
        metas = []

        for f in files:
            filename = f["filename"]
            raw = self._read_file_bytes(filename, f["bytes"])
            chunks = self._chunk_text(raw)

            for i, ch in enumerate(chunks):
                chunk_id = f"{filename}::chunk_{i}"
                ids.append(chunk_id)
                docs.append(ch)
                metas.append({"source": filename, "chunk_index": i})

        if not docs:
            return {"added_chunks": 0, "notes": "No readable text found in uploaded files."}

        emb = self.embedder.encode(docs, normalize_embeddings=True)
        emb = np.array(emb).tolist()

        self.col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=emb
        )

        return {"added_chunks": len(docs), "sources": sorted(list({m["source"] for m in metas}))}

    # ---------- retrieval ----------
    def query(self, q: str, k: int = 4) -> List[RagHit]:
        q = (q or "").strip()
        if not q:
            return []

        qemb = self.embedder.encode([q], normalize_embeddings=True)
        qemb = np.array(qemb).tolist()

        # ✅ FIX: do NOT include "ids" here (Chroma returns ids automatically)
        res = self.col.query(
            query_embeddings=qemb,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]  # ✅ ids still available here

        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            hits.append(RagHit(
                text=doc,
                source=(meta or {}).get("source", "unknown"),
                chunk_id=cid,
                distance=float(dist) if dist is not None else 0.0
            ))

        return hits

    def clear(self) -> Dict[str, Any]:
        self.client.delete_collection(self.collection_name)
        self.col = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return {"ok": True}

    def stats(self) -> Dict[str, Any]:
        try:
            n = self.col.count()
        except Exception:
            n = None
        return {"collection": self.collection_name, "chunks": n, "persist_dir": self.persist_dir}