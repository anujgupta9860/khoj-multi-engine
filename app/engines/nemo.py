"""NVIDIA NeMo Retriever engine (docs/multi-engine-plan.md).

Unlike OpenRAG and RAGFlow there is no ingestion service on the
hosted path, so this adapter does the pipeline itself:

  extract text (agent-style: stdlib DOCX, pypdf PDF, plain decode)
  -> chunk 1000/200 -> embed with the NeMo Retriever embedding NIM
  (nvidia/llama-nemotron-embed-1b-v2) -> upsert into Milvus
  -> search = embed query (input_type=query) + Milvus COSINE search
     filtered to the caller's documents, optional rerank NIM.

Milvus is spoken to over its REST v2 API, so the web image needs no
extra client library. The collection is created quick-style
(id + vector) with dynamic fields carrying doc_id / scope_key /
chunk_text; every search additionally post-filters by doc_id here.

The NIM base URL is the hosted API by default; self-hosted NIMs
later only change NVIDIA_BASE_URL.
"""

from __future__ import annotations

import html
import io
import os
import re
import zipfile

import requests

from .base import Engine, EngineError


def _env(name, default=""):
    return os.environ.get(name, default)


NVIDIA_BASE_URL = _env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = _env("NVIDIA_EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
RERANK_MODEL = _env("NVIDIA_RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
EMBED_DIM = int(_env("NEMO_EMBED_DIM", "1024"))
MILVUS_URL = _env("MILVUS_URL", "http://milvus:19530")
COLLECTION = _env("NEMO_COLLECTION", "khoj_chunks")
MIN_SCORE = float(_env("NEMO_MIN_SCORE", "0.35"))
USE_RERANK = _env("NVIDIA_RERANK", "1") not in ("0", "false", "")

_TAG_RE = re.compile(r"<[^>]+>")


def _extract_docx(raw: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception:
        return None
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    return html.unescape(_TAG_RE.sub("", xml))


def _extract_pdf(raw: bytes) -> str | None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return None


def extract_text(filename: str, raw: bytes) -> str | None:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".docx":
        return _extract_docx(raw)
    if ext == ".pdf":
        return _extract_pdf(raw)
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return None
    if ext in (".html", ".htm"):
        text = re.sub(r"(?i)</(p|div|li|tr|h[1-6])>|<br\s*/?>", "\n", text)
        text = html.unescape(_TAG_RE.sub(" ", text))
    return text


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list:
    """Paragraph-aware packing to ~size chars; a paragraph longer
    than size is window-split with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(para) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            step = max(size - overlap, 1)
            for i in range(0, len(para), step):
                chunks.append(para[i:i + size])
            continue
        if buf and len(buf) + len(para) + 1 > size:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


class NeMoEngine(Engine):
    name = "nemo"

    def __init__(self, http=None, base_url: str | None = None,
                 milvus_url: str | None = None, api_key: str | None = None):
        self.http = http or requests
        self.base = (base_url or NVIDIA_BASE_URL).rstrip("/")
        self.milvus = (milvus_url or MILVUS_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else _env("NVIDIA_API_KEY")
        self._collection_ready = False

    # -- NIM calls -----------------------------------------------------
    def _embed(self, texts: list, input_type: str) -> list:
        if not self.api_key:
            raise EngineError("NVIDIA_API_KEY is not configured")
        out = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            resp = self.http.post(
                f"{self.base}/embeddings",
                json={"model": EMBED_MODEL, "input": batch,
                      "input_type": input_type, "truncate": "END"},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60,
            )
            if resp.status_code != 200:
                raise EngineError(f"NIM embeddings: HTTP {resp.status_code} {resp.text[:200]}")
            data = resp.json().get("data", [])
            vecs = [d.get("embedding", []) for d in sorted(data, key=lambda d: d.get("index", 0))]
            for vec in vecs:
                if len(vec) != EMBED_DIM:
                    raise EngineError(
                        f"NIM embedding dim {len(vec)} != NEMO_EMBED_DIM {EMBED_DIM}; "
                        "set NEMO_EMBED_DIM to the model's dimension"
                    )
            out.extend(vecs)
        return out

    def _rerank(self, query: str, chunks: list) -> list:
        try:
            resp = self.http.post(
                f"{self.base}/ranking",
                json={"model": RERANK_MODEL, "query": {"text": query},
                      "passages": [{"text": c["text"]} for c in chunks],
                      "truncate": "END"},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            if resp.status_code != 200:
                return chunks
            rankings = resp.json().get("rankings", [])
            if not rankings:
                return chunks
            return [chunks[r["index"]] for r in sorted(rankings, key=lambda r: -r.get("logit", 0))
                    if 0 <= r.get("index", -1) < len(chunks)]
        except Exception:
            return chunks

    # -- Milvus calls ----------------------------------------------------
    def _mv(self, path: str, payload: dict):
        resp = self.http.post(f"{self.milvus}/v2/vectordb/{path}", json=payload, timeout=30)
        if resp.status_code != 200:
            raise EngineError(f"Milvus {path}: HTTP {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        if body.get("code") not in (0, 200):
            raise EngineError(f"Milvus {path}: {body.get('message') or body}")
        return body.get("data")

    def _ensure_collection(self):
        if self._collection_ready:
            return
        has = self._mv("collections/has", {"collectionName": COLLECTION})
        if not (isinstance(has, dict) and has.get("has")):
            self._mv("collections/create", {
                "collectionName": COLLECTION,
                "dimension": EMBED_DIM,
                "metricType": "COSINE",
            })
        self._collection_ready = True

    # -- ingest ----------------------------------------------------------
    def ingest(self, *, filename, data, mime, scope, scope_key, doc_id):
        text = extract_text(filename, data)
        if not text or not text.strip():
            raise EngineError(f"No extractable text in {filename} for the NeMo engine")
        chunks = chunk_text(text)
        if not chunks:
            raise EngineError(f"No chunks produced from {filename}")
        self._ensure_collection()
        vectors = self._embed(chunks, "passage")
        rows = [
            {"vector": vec, "doc_id": doc_id, "scope_key": scope_key,
             "chunk_text": chunk}
            for vec, chunk in zip(vectors, chunks)
        ]
        self._mv("entities/insert", {"collectionName": COLLECTION, "data": rows})
        return "", doc_id

    def indexed_doc_ids(self, records):
        if not records:
            return set()
        self._ensure_collection()
        ready = set()
        for rec in records:
            doc_id = rec.get("docId", "")
            data = self._mv("entities/query", {
                "collectionName": COLLECTION,
                "filter": f'doc_id == "{doc_id}"',
                "outputFields": ["doc_id"],
                "limit": 1,
            })
            if data:
                ready.add(doc_id)
        return ready

    # -- search ----------------------------------------------------------
    def search_chunks(self, query, records, limit=40, doc_ids=None):
        if not records:
            return []
        wanted = {rec.get("docId", ""): rec for rec in records}
        if doc_ids:
            wanted = {d: wanted[d] for d in doc_ids if d in wanted}
            if not wanted:
                return []
        self._ensure_collection()
        vec = self._embed([query], "query")[0]
        id_list = ", ".join(f'"{d}"' for d in wanted)
        data = self._mv("entities/search", {
            "collectionName": COLLECTION,
            "data": [vec],
            "filter": f"doc_id in [{id_list}]",
            "limit": limit,
            "outputFields": ["doc_id", "chunk_text"],
        }) or []
        out = []
        for hit in data:
            rec = wanted.get(hit.get("doc_id", ""))
            if rec is None:
                continue  # another corpus's document — never returned
            score = hit.get("distance", 0)
            if score < MIN_SCORE:
                continue
            out.append({
                "doc_id": rec.get("docId", ""),
                "filename": rec.get("filename", ""),
                "text": hit.get("chunk_text", ""),
                "score": score,
            })
        if USE_RERANK and self.api_key and len(out) > 1:
            out = self._rerank(query, out)
        return out

    # -- delete ------------------------------------------------------------
    def delete_document(self, record, scope):
        self._ensure_collection()
        self._mv("entities/delete", {
            "collectionName": COLLECTION,
            "filter": f'doc_id == "{record.get("docId", "")}"',
        })
