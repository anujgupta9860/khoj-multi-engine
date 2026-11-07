"""Search-only engine client for the A2A agent.

The agent is a separate, stateless service: it never ingests or
deletes, it only searches the corpus the web service authorized
(/api/agent/corpus). This module mirrors app/engines' search half
for KHOJ_ENGINE=openrag|ragflow so the agent image stays standalone.

The corpus map it receives is keyed by the engine's per-document
key (OpenRAG filename / RAGFlow document id / doc id) — the key IS
the corpus boundary, and every returned chunk is post-filtered
against it by the caller.
"""

from __future__ import annotations

import os

import requests


def _env(name, default=""):
    return os.environ.get(name, default)

ENGINE = os.environ.get("KHOJ_ENGINE", "openrag").strip().lower()
OPENRAG_URL = os.environ.get("OPENRAG_URL", "http://openrag-backend:8000")
RAGFLOW_URL = os.environ.get("RAGFLOW_URL", "http://ragflow:9380")
RAGFLOW_API_KEY = _env("RAGFLOW_API_KEY")
RAGFLOW_MIN_SIMILARITY = float(os.environ.get("RAGFLOW_MIN_SIMILARITY", "0.2"))

_rf_datasets: dict[str, str] = {}


def _rf_dataset_id(scope_key: str) -> str | None:
    if scope_key in _rf_datasets:
        return _rf_datasets[scope_key]
    name = "khoj-" + scope_key.replace(":", "-").replace("/", "-")
    try:
        resp = requests.get(
            f"{RAGFLOW_URL}/api/v1/datasets",
            params={"name": name, "page": 1, "page_size": 10},
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        for ds in resp.json().get("data") or []:
            if ds.get("name") == name:
                _rf_datasets[scope_key] = ds["id"]
                return ds["id"]
    except Exception:
        return None
    return None


def search_chunks(query: str, corpus: dict, limit: int = 30) -> list:
    """Normalized chunks: {"key", "text", "score"} — key is the
    corpus-map key of the document the chunk came from."""
    if not corpus:
        return []
    if ENGINE == "ragflow":
        return _ragflow_search(query, corpus, limit)
    return _openrag_search(query, corpus, limit)


def _openrag_search(query: str, corpus: dict, limit: int) -> list:
    resp = requests.post(
        f"{OPENRAG_URL}/search",
        json={"query": query, "filters": {"filename": sorted(corpus)}, "limit": limit},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Search backend: {resp.text[:200]}")
    data = resp.json()
    chunks = data if isinstance(data, list) else data.get("results", [])
    return [
        {"key": c.get("filename", ""), "text": c.get("text", ""), "score": c.get("score", 0)}
        for c in chunks
    ]


def _ragflow_search(query: str, corpus: dict, limit: int) -> list:
    dataset_ids = []
    for info in corpus.values():
        ds = _rf_dataset_id(info.get("scopeKey", ""))
        if ds and ds not in dataset_ids:
            dataset_ids.append(ds)
    if not dataset_ids:
        return []
    resp = requests.post(
        f"{RAGFLOW_URL}/api/v1/retrieval",
        json={
            "question": query,
            "dataset_ids": dataset_ids,
            "page": 1,
            "page_size": limit,
            "similarity_threshold": RAGFLOW_MIN_SIMILARITY,
        },
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Search backend: {resp.text[:200]}")
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(f"Search backend: {body.get('message')}")
    by_ref = {info.get("engineRef"): key for key, info in corpus.items() if info.get("engineRef")}
    out = []
    for chunk in (body.get("data") or {}).get("chunks", []):
        key = by_ref.get(chunk.get("document_id", ""))
        if key is None:
            continue
        out.append({
            "key": key,
            "text": chunk.get("content", ""),
            "score": chunk.get("similarity", 0),
        })
    return out
