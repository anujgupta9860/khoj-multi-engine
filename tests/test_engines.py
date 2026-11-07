"""Engine contract tests: OpenRAG (verbatim-behavior) and RAGFlow
(dataset-per-scope) against fake HTTP transports. No network.

Run: python tests/test_engines.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engines.openrag import OpenRAGEngine  # noqa: E402
from app.engines.ragflow import RAGFlowEngine, dataset_name  # noqa: E402
from app.engines.base import EngineError  # noqa: E402


class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body


class FakeHTTP:
    """Records calls; routes by (method, path-suffix) to handlers."""

    def __init__(self):
        self.calls = []
        self.routes = {}

    def on(self, method, suffix, fn):
        self.routes[(method, suffix)] = fn
        return self

    def _dispatch(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for (m, suffix), fn in self.routes.items():
            if m == method and url.endswith(suffix):
                return fn(url, kw)
        raise AssertionError(f"no fake route for {method} {url}")

    def get(self, url, **kw):
        return self._dispatch("GET", url, **kw)

    def post(self, url, **kw):
        return self._dispatch("POST", url, **kw)

    def delete(self, url, **kw):
        return self._dispatch("DELETE", url, **kw)

    def request(self, method, url, **kw):
        return self._dispatch(method, url, **kw)


REC = {"docId": "doc-abc", "filename": "w2.pdf", "scopeKey": "user:u1"}


def test_openrag_search_floor_and_scope():
    http = FakeHTTP()
    http.on("POST", "/search", lambda url, kw: Resp(200, [
        {"filename": "doc-abc--w2.pdf", "text": "real", "score": 1.9},
        {"filename": "doc-abc--w2.pdf", "text": "junk", "score": 1.1},
        {"filename": "doc-zzz--other.pdf", "text": "alien", "score": 2.5},
    ]))
    eng = OpenRAGEngine(http=http, base_url="http://or")
    chunks = eng.search_chunks("q", [REC])
    assert [c["text"] for c in chunks] == ["real"], chunks
    assert chunks[0]["doc_id"] == "doc-abc"
    payload = http.calls[0][2]["json"]
    assert payload["filters"]["filename"] == ["doc-abc--w2.pdf"], payload
    print("openrag search ok")


def test_openrag_delete_no_chunks_is_success():
    http = FakeHTTP()
    http.on("POST", "/documents/delete-by-filename", lambda url, kw: Resp(404, {
        "success": False, "deleted_chunks": 0,
        "error": "No matching document chunks were deleted.",
    }))
    eng = OpenRAGEngine(http=http, base_url="http://or")
    eng.delete_document(REC, "users/u1")  # must not raise
    print("openrag delete no-chunks ok")


def test_openrag_delete_real_failure_raises():
    http = FakeHTTP()
    http.on("POST", "/documents/delete-by-filename",
            lambda url, kw: Resp(500, {"success": False, "error": "boom"}, text="boom"))
    eng = OpenRAGEngine(http=http, base_url="http://or")
    try:
        eng.delete_document(REC, "users/u1")
    except EngineError:
        print("openrag delete failure ok")
        return
    raise AssertionError("expected EngineError")


def test_ragflow_dataset_name():
    assert dataset_name("org:khoj") == "khoj-org-khoj"
    assert dataset_name("user:u1") == "khoj-user-u1"
    print("ragflow dataset name ok")


def _ragflow_http():
    http = FakeHTTP()
    http.on("GET", "/api/v1/datasets?name=khoj-user-u1&page=1&page_size=10",
            lambda url, kw: Resp(200, {"code": 0, "data": []}))
    http.on("POST", "/api/v1/datasets",
            lambda url, kw: Resp(200, {"code": 0, "data": {"id": "ds1"}}))
    http.on("POST", "/api/v1/datasets/ds1/documents",
            lambda url, kw: Resp(200, {"code": 0, "data": [{"id": "rf-doc-1"}]}))
    http.on("POST", "/api/v1/datasets/ds1/chunks",
            lambda url, kw: Resp(200, {"code": 0, "data": True}))
    http.on("GET", "/api/v1/datasets/ds1/documents?page=1&page_size=100",
            lambda url, kw: Resp(200, {"code": 0, "data": {"docs": [
                {"id": "rf-doc-1", "name": "w2.pdf", "run": "DONE", "chunk_count": 7},
            ]}}))
    http.on("POST", "/api/v1/retrieval", lambda url, kw: Resp(200, {"code": 0, "data": {
        "chunks": [
            {"content": "hit", "document_id": "rf-doc-1",
             "document_keyword": "w2.pdf", "similarity": 0.87},
            {"content": "alien", "document_id": "rf-doc-9",
             "document_keyword": "x.pdf", "similarity": 0.99},
        ]}}))
    http.on("DELETE", "/api/v1/datasets/ds1/documents",
            lambda url, kw: Resp(200, {"code": 0, "data": True}))
    return http


def test_ragflow_end_to_end():
    http = _ragflow_http()
    eng = RAGFlowEngine(http=http, base_url="http://rf", api_key="k")
    task_id, ref = eng.ingest(filename="w2.pdf", data=b"%PDF", mime="application/pdf",
                              scope="users/u1", scope_key="user:u1", doc_id="doc-abc")
    assert ref == "rf-doc-1" and task_id == ""
    rec = dict(REC, engineRef=ref)
    assert eng.indexed_doc_ids([rec]) == {"doc-abc"}
    assert eng.index_meta([rec])["doc-abc"]["chunks"] == 7
    chunks = eng.search_chunks("wages", [rec])
    assert [c["text"] for c in chunks] == ["hit"], chunks  # alien doc filtered out
    payload = [kw for m, u, kw in http.calls if u.endswith("/api/v1/retrieval")][0]["json"]
    assert payload["dataset_ids"] == ["ds1"], payload
    eng.delete_document(rec, "users/u1")
    print("ragflow end-to-end ok")


def test_ragflow_ingest_starts_parsing():
    http = _ragflow_http()
    eng = RAGFlowEngine(http=http, base_url="http://rf", api_key="k")
    eng.ingest(filename="w2.pdf", data=b"%PDF", mime="application/pdf",
               scope="users/u1", scope_key="user:u1", doc_id="doc-abc")
    parse_calls = [kw for m, u, kw in http.calls if u.endswith("/chunks")]
    assert parse_calls and parse_calls[0]["json"] == {"document_ids": ["rf-doc-1"]}
    print("ragflow parse-start ok")


if __name__ == "__main__":
    test_openrag_search_floor_and_scope()
    test_openrag_delete_no_chunks_is_success()
    test_openrag_delete_real_failure_raises()
    test_ragflow_dataset_name()
    test_ragflow_end_to_end()
    test_ragflow_ingest_starts_parsing()
    print("ALL ENGINE TESTS PASSED")
