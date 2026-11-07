"""RAGFlow engine (docs/multi-engine-plan.md).

Scoping: ONE DATASET PER SCOPE — a corpus is a RAGFlow dataset, so
isolation is structural (retrieval only ever names the caller's
dataset plus the platform-org/global dataset) instead of the
filename-prefix discipline OpenRAG needs. The record's engineRef is
the RAGFlow document id inside its scope's dataset.

API notes (RAGFlow HTTP API): every response is an envelope
{code, message, data} with code 0 on success; auth is
'Authorization: Bearer <api key>'; parsing is explicit — an uploaded
document does nothing until POST /datasets/{id}/chunks starts it;
document run states are UNSTART/RUNNING/CANCEL/DONE/FAIL.
"""

from __future__ import annotations

import io
import os
import time

import requests


def _env(name, default=""):
    return os.environ.get(name, default)

from .base import Engine, EngineError

RAGFLOW_URL = os.environ.get("RAGFLOW_URL", "http://ragflow:9380")
RAGFLOW_API_KEY = _env("RAGFLOW_API_KEY")
MIN_SIMILARITY = float(os.environ.get("RAGFLOW_MIN_SIMILARITY", "0.2"))
EMBEDDING_MODEL = os.environ.get("RAGFLOW_EMBEDDING_MODEL", "")


def dataset_name(scope_key: str) -> str:
    """'org:khoj' -> 'khoj-org-khoj'; 'user:<uid>' -> 'khoj-user-<uid>'."""
    return "khoj-" + scope_key.replace(":", "-").replace("/", "-")


class RAGFlowEngine(Engine):
    name = "ragflow"

    def __init__(self, http=None, base_url: str | None = None,
                 api_key: str | None = None):
        self.http = http or requests
        self.base = (base_url or RAGFLOW_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else RAGFLOW_API_KEY
        self._datasets: dict[str, str] = {}      # scope_key -> dataset_id
        self._docs_cache: dict[str, tuple] = {}  # dataset_id -> (at, docs)

    # -- plumbing ------------------------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _call(self, method: str, path: str, **kw):
        resp = self.http.request(
            method, f"{self.base}{path}", headers=self._headers(), timeout=30, **kw
        )
        if resp.status_code != 200:
            raise EngineError(f"RAGFlow {method} {path}: HTTP {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        if body.get("code") != 0:
            raise EngineError(f"RAGFlow {method} {path}: {body.get('message') or body}")
        return body.get("data")

    def _dataset_id(self, scope_key: str, create: bool = False) -> str | None:
        if scope_key in self._datasets:
            return self._datasets[scope_key]
        name = dataset_name(scope_key)
        data = self._call("GET", f"/api/v1/datasets?name={name}&page=1&page_size=10")
        for ds in data or []:
            if ds.get("name") == name:
                self._datasets[scope_key] = ds["id"]
                return ds["id"]
        if not create:
            return None
        # RAGFlow v1.0 dataset-create schema: no chunk_method field
        # (rejected as an extra input, live finding 2026-11-07); the
        # general parser is selected with parse_type/parser_id.
        payload = {"name": name, "parse_type": 1, "parser_id": "general"}
        if EMBEDDING_MODEL:
            payload["embedding_model"] = EMBEDDING_MODEL
        data = self._call("POST", "/api/v1/datasets", json=payload)
        self._datasets[scope_key] = data["id"]
        return data["id"]

    def _documents(self, dataset_id: str) -> list:
        at, docs = self._docs_cache.get(dataset_id, (0.0, []))
        now = time.monotonic()
        if now - at < 5:
            return docs
        data = self._call(
            "GET", f"/api/v1/datasets/{dataset_id}/documents?page=1&page_size=100"
        )
        docs = (data or {}).get("docs", []) if isinstance(data, dict) else []
        self._docs_cache[dataset_id] = (now, docs)
        return docs

    # -- ingest --------------------------------------------------------
    def ingest(self, *, filename, data, mime, scope, scope_key, doc_id):
        ds = self._dataset_id(scope_key, create=True)
        resp = self.http.post(
            f"{self.base}/api/v1/datasets/{ds}/documents",
            headers=self._headers(),
            files={"file": (filename, io.BytesIO(data), mime)},
            timeout=120,
        )
        if resp.status_code != 200:
            raise EngineError(f"RAGFlow upload: HTTP {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        if body.get("code") != 0:
            raise EngineError(f"RAGFlow upload: {body.get('message') or body}")
        docs = body.get("data") or []
        if not docs:
            raise EngineError("RAGFlow upload: no document returned")
        ref = docs[0]["id"]
        # Upload alone does not parse — start parsing explicitly.
        self._call("POST", f"/api/v1/datasets/{ds}/chunks",
                   json={"document_ids": [ref]})
        self._docs_cache.pop(ds, None)
        return "", ref

    def doc_error(self, record):
        ref = record.get("engineRef", "")
        ds = self._dataset_id(record.get("scopeKey", ""))
        if not ref or not ds:
            return None
        try:
            data = self._call(
                "GET", f"/api/v1/datasets/{ds}/documents?id={ref}&page=1&page_size=1"
            )
        except Exception:
            return None
        docs = (data or {}).get("docs", []) if isinstance(data, dict) else []
        for doc in docs:
            if doc.get("id") == ref:
                run = str(doc.get("run", "")).upper()
                if run == "FAIL":
                    return str(doc.get("progress_msg") or "parsing failed")[-200:]
                if run == "CANCEL":
                    return "parsing cancelled"
                return None
        return None

    def indexed_doc_ids(self, records):
        by_scope: dict[str, list] = {}
        for rec in records:
            by_scope.setdefault(rec.get("scopeKey", ""), []).append(rec)
        ready = set()
        for scope_key, recs in by_scope.items():
            ds = self._dataset_id(scope_key)
            if not ds:
                continue
            docs = {d.get("id"): d for d in self._documents(ds)}
            by_name = {d.get("name"): d for d in docs.values()}
            for rec in recs:
                doc = docs.get(rec.get("engineRef", "")) or by_name.get(rec.get("filename", ""))
                if doc and str(doc.get("run", "")).upper() == "DONE":
                    ready.add(rec.get("docId", ""))
        return ready

    def index_meta(self, records):
        by_scope: dict[str, list] = {}
        for rec in records:
            by_scope.setdefault(rec.get("scopeKey", ""), []).append(rec)
        out = {}
        for scope_key, recs in by_scope.items():
            ds = self._dataset_id(scope_key)
            if not ds:
                continue
            docs = {d.get("id"): d for d in self._documents(ds)}
            by_name = {d.get("name"): d for d in docs.values()}
            for rec in recs:
                doc = docs.get(rec.get("engineRef", "")) or by_name.get(rec.get("filename", ""))
                if doc and str(doc.get("run", "")).upper() == "DONE":
                    out[rec.get("docId", "")] = {
                        "chunks": doc.get("chunk_count"),
                        "indexed_time": None,
                    }
        return out

    # -- search --------------------------------------------------------
    def search_chunks(self, query, records, limit=40, doc_ids=None):
        if not records:
            return []
        scope_keys = []
        for rec in records:
            sk = rec.get("scopeKey", "")
            if sk not in scope_keys:
                scope_keys.append(sk)
        dataset_ids = []
        for sk in scope_keys:
            ds = self._dataset_id(sk)
            if ds:
                dataset_ids.append(ds)
        if not dataset_ids:
            return []
        by_ref = {rec.get("engineRef", ""): rec for rec in records if rec.get("engineRef")}
        payload = {
            "question": query,
            "dataset_ids": dataset_ids,
            "page": 1,
            "page_size": limit,
            "similarity_threshold": MIN_SIMILARITY,
        }
        if doc_ids:
            refs = [rec.get("engineRef") for rec in records
                    if rec.get("docId") in doc_ids and rec.get("engineRef")]
            if not refs:
                return []
            payload["document_ids"] = refs
        data = self._call("POST", "/api/v1/retrieval", json=payload)
        chunks = (data or {}).get("chunks", []) if isinstance(data, dict) else []
        out = []
        for chunk in chunks:
            rec = by_ref.get(chunk.get("document_id", ""))
            if rec is None:
                continue  # not one of the caller's documents — never returned
            if doc_ids and rec.get("docId") not in doc_ids:
                continue
            out.append({
                "doc_id": rec.get("docId", ""),
                "filename": rec.get("filename", ""),
                "text": chunk.get("content", ""),
                "score": chunk.get("similarity", 0),
            })
        return out

    def corpus_key(self, record):
        return record.get("engineRef") or record.get("docId", "")

    # -- delete --------------------------------------------------------
    def delete_document(self, record, scope):
        ref = record.get("engineRef", "")
        ds = self._dataset_id(record.get("scopeKey", ""))
        if not ref or not ds:
            return  # nothing was ever indexed under this record
        self._call("DELETE", f"/api/v1/datasets/{ds}/documents", json={"ids": [ref]})
        self._docs_cache.pop(ds, None)
