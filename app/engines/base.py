"""Retrieval engine contract for Khoj.

The product layer (web app, MCP, agent) talks to exactly one engine,
chosen per deployment with KHOJ_ENGINE=openrag|ragflow|nemo. An
engine owns everything about how documents get indexed and searched;
Khoj's own database stays the system of record for corpora (upload
records), and every engine call is scoped by those records.

Records are the upload dicts from the org service: docId, filename,
scopeKey ("user:<uid>" / "org:<orgId>"), plus engineTask / engineRef
written at ingest time. Chunks returned by search are normalized to
{"doc_id", "filename", "text", "score"} so callers never see an
engine's native shapes.
"""

from __future__ import annotations


class EngineError(Exception):
    """A hard engine failure. Callers surface it as a 502 and, for
    deletes, treat it as 'nothing was removed'."""


class Engine:
    name = "base"

    # -- ingest ------------------------------------------------------
    def ingest(self, *, filename: str, data: bytes, mime: str,
               scope: str, scope_key: str, doc_id: str) -> tuple[str, str]:
        """Index one document. Returns (task_id, engine_ref): task_id
        is whatever status handle the engine has ('' when none) and
        engine_ref is the engine-side document identifier the record
        should carry for later delete/search scoping ('' when the
        engine addresses documents by name, like OpenRAG)."""
        raise NotImplementedError

    def doc_error(self, record: dict) -> str | None:
        """Error text when this record's ingest failed; None when it
        succeeded, is still running, or cannot be checked."""
        return None

    def indexed_doc_ids(self, records: list) -> set:
        """The subset of the given records' docIds that the engine
        has fully indexed (status 'ready')."""
        raise NotImplementedError

    def index_meta(self, records: list) -> dict:
        """Per-document index metadata for the admin views:
        docId -> {"chunks": int|None, "indexed_time": Any}. Only
        fully-indexed documents appear. Default: presence only."""
        return {doc_id: {} for doc_id in self.indexed_doc_ids(records)}

    # -- search ------------------------------------------------------
    def search_chunks(self, query: str, records: list, limit: int = 40,
                      doc_ids: list | None = None) -> list:
        """Scoped chunk search across exactly the given corpus
        records (own + global, already authorized by the caller).
        doc_ids optionally narrows to specific documents (the
        Knowledge-tab per-document search). Returns normalized chunk
        dicts, best first, with the engine's own relevance floor
        already applied."""
        raise NotImplementedError

    def corpus_key(self, record: dict) -> str:
        """The per-document key the agent's corpus map uses for this
        record (engine-native document identity)."""
        return record.get("docId", "")

    # -- delete / per-document handles -------------------------------
    def delete_document(self, record: dict, scope: str) -> None:
        """Remove the document's chunks from the index. Raises
        EngineError on a real index failure; 'already absent' is
        success (a still-indexing document has no chunks)."""
        raise NotImplementedError

    def after_upload(self, record: dict, scope: str) -> None:
        """Best-effort per-document handle creation after ingest
        (OpenRAG's knowledge filter). No-op where the engine scopes
        search by document id natively."""

    def remove_doc_handle(self, record: dict, scope: str) -> None:
        """Best-effort counterpart of after_upload, on delete."""
