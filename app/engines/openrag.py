"""OpenRAG engine — the original Khoj backend, moved verbatim
behind the engine contract (docs/multi-engine-plan.md).

Scoping: documents live in OpenRAG under `{doc_id}--{filename}`;
searches carry a server-side filename filter over exactly the
corpus's names, and results are post-filtered by doc_id as a second
line of defence. Routes are the backend's UNPREFIXED ones — the
/api/* prefix exists only on its frontend proxy.
"""

from __future__ import annotations

import io
import os
import time

import requests

from .base import Engine, EngineError

OPENRAG_URL = os.environ.get("OPENRAG_URL", "http://openrag-backend:8000")
MIN_SCORE = float(os.environ.get("KHOJ_MIN_SCORE", "1.5"))


def openrag_filename(doc_id: str, filename: str) -> str:
    return f"{doc_id}--{filename}"


class OpenRAGEngine(Engine):
    name = "openrag"

    def __init__(self, http=None, base_url: str | None = None):
        self.http = http or requests
        self.base = (base_url or OPENRAG_URL).rstrip("/")
        self._files_cache = {"at": 0.0, "names": {}}

    # -- ingest ------------------------------------------------------
    def ingest(self, *, filename, data, mime, scope, scope_key, doc_id):
        or_name = openrag_filename(doc_id, filename)
        resp = self.http.post(
            f"{self.base}/router/upload_ingest",
            files={"file": (or_name, io.BytesIO(data), mime)},
            timeout=120,
        )
        if resp.status_code not in (200, 202):
            raise EngineError(f"Ingestion trigger: {resp.text[:300]}")
        task_id = ""
        try:
            task_id = resp.json().get("task_id", "")
        except Exception:
            pass
        return task_id, ""

    def doc_error(self, record):
        task_id = record.get("engineTask") or record.get("openragTask", "")
        if not task_id:
            return None
        for path in (f"/tasks/{task_id}", f"/v1/tasks/{task_id}"):
            try:
                resp = self.http.get(f"{self.base}{path}", timeout=10)
            except Exception:
                return None
            if resp.status_code == 404:
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            status = str(data.get("status", "")).lower()
            if status in ("failed", "error", "failure"):
                return str(
                    data.get("error") or data.get("message") or "ingestion failed"
                )[:200]
            return None
        return None

    def _files(self) -> dict:
        now = time.monotonic()
        if now - self._files_cache["at"] < 5:
            return self._files_cache["names"]
        resp = self.http.get(f"{self.base}/files", timeout=15)
        if resp.status_code != 200:
            raise EngineError(f"List backend: {resp.text[:300]}")
        data = resp.json()
        entries = data.get("files", data) if isinstance(data, dict) else data
        names = {}
        for entry in entries or []:
            name = entry.get("filename") or entry.get("title") or ""
            if name:
                names[name] = entry
        self._files_cache.update(at=now, names=names)
        return names

    def indexed_doc_ids(self, records):
        files = self._files()
        return {
            rec.get("docId", "")
            for rec in records
            if openrag_filename(rec.get("docId", ""), rec.get("filename", ""))
            in files
        }

    def index_meta(self, records):
        files = self._files()
        out = {}
        for rec in records:
            entry = files.get(
                openrag_filename(rec.get("docId", ""), rec.get("filename", ""))
            )
            if entry is not None:
                out[rec.get("docId", "")] = {
                    "chunks": entry.get("chunks"),
                    "indexed_time": entry.get("indexed_time"),
                }
        return out

    # -- search ------------------------------------------------------
    def search_chunks(self, query, records, limit=40, doc_ids=None):
        by_name = {}
        for rec in records:
            if doc_ids and rec.get("docId") not in doc_ids:
                continue
            by_name[openrag_filename(rec.get("docId", ""), rec.get("filename", ""))] = rec
        if not by_name:
            return []
        resp = self.http.post(
            f"{self.base}/search",
            json={"query": query, "filters": {"filename": sorted(by_name)},
                  "limit": limit},
            timeout=30,
        )
        if resp.status_code != 200:
            raise EngineError(f"Search backend: {resp.text[:300]}")
        data = resp.json()
        chunks = data if isinstance(data, list) else data.get("results", [])
        out = []
        for chunk in chunks:
            name = chunk.get("filename", "")
            rec = by_name.get(name)
            if rec is None:
                continue  # another corpus's document — never returned
            score = chunk.get("score", 0)
            if score < MIN_SCORE:
                continue
            out.append({
                "doc_id": rec.get("docId", ""),
                "filename": rec.get("filename", ""),
                "text": chunk.get("text", ""),
                "score": score,
            })
        return out

    def corpus_key(self, record):
        return openrag_filename(record.get("docId", ""), record.get("filename", ""))

    # -- delete / per-document handles -------------------------------
    def delete_document(self, record, scope):
        or_name = openrag_filename(record.get("docId", ""), record.get("filename", ""))
        try:
            resp = self.http.post(
                f"{self.base}/documents/delete-by-filename",
                json={"filename": or_name},
                timeout=30,
            )
            try:
                payload = resp.json()
            except Exception:
                payload = {}
        except Exception as exc:
            raise EngineError(f"Index delete unreachable, nothing was removed: {exc}")
        if not isinstance(payload, dict):
            payload = {}
        # OpenRAG reports "no matching chunks" as a failure — for a
        # still-indexing or failed document that IS the desired end
        # state (nothing in the index), not an error.
        no_chunks = payload.get("deleted_chunks") == 0 and (
            "No matching document chunks" in str(payload.get("error") or "")
            or "No matching document chunks" in str(payload.get("message") or "")
        )
        index_ok = (
            resp.status_code == 200 and payload.get("success") is not False
        ) or no_chunks
        if not index_ok:
            raise EngineError(
                f"Index delete failed, nothing was removed: {resp.text[:200]}"
            )
        self._files_cache["at"] = 0.0

    def after_upload(self, record, scope):
        """Create the per-document knowledge filter (best-effort) —
        the same payload the web app has always sent."""
        or_name = openrag_filename(record.get("docId", ""), record.get("filename", ""))
        filename = record.get("filename", "")
        display = os.path.splitext(filename)[0] or filename
        query_data = {
            "query": "",
            "filters": {
                "data_sources": [or_name],
                "document_types": ["*"],
                "owners": ["*"],
                "connector_types": ["*"],
            },
            "limit": 10,
            "scoreThreshold": 0,
            "color": "green",
            "icon": "file",
        }
        try:
            self.http.post(
                f"{self.base}/knowledge-filter",
                json={
                    "name": display,
                    "description": f"Uploaded via Khoj: {filename}",
                    "queryData": query_data,
                },
                timeout=15,
            )
        except Exception:
            pass

    def remove_doc_handle(self, record, scope):
        """Delete the document's knowledge filter (best-effort).
        Filter dicts are extracted defensively from whatever nesting
        the search endpoint returns, and matched by data_sources or
        by the exact 'Uploaded via Khoj' description."""
        or_name = openrag_filename(record.get("docId", ""), record.get("filename", ""))
        filename = record.get("filename", "")
        display = os.path.splitext(filename)[0] or filename
        try:
            resp = self.http.post(
                f"{self.base}/knowledge-filter/search",
                json={"query": display, "limit": 25},
                timeout=15,
            )
            if resp.status_code != 200:
                return
            candidates = []
            stack = [resp.json()]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if "id" in cur and ("queryData" in cur or "name" in cur):
                        candidates.append(cur)
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
        except Exception:
            return
        for entry in candidates:
            qd = entry.get("queryData") or entry.get("query_data") or {}
            sources = ((qd.get("filters") or {}).get("data_sources")) or []
            if or_name in sources or entry.get("description") == (
                f"Uploaded via Khoj: {filename}"
            ):
                try:
                    self.http.delete(
                        f"{self.base}/knowledge-filter/{entry['id']}", timeout=15
                    )
                except Exception:
                    pass
