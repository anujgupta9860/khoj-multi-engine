# Multi-engine plan — OpenRAG / RAGFlow / NVIDIA NeMo Retriever

One Khoj product, three retrieval engines. The web app, auth, orgs,
MCP server, A2A agent and Jira integration stay identical; the engine
is chosen per deployment with `KHOJ_ENGINE=openrag|ragflow|nemo`.
Three deployments run side by side in the same GKE cluster so the
same corpus and the same questions can be compared across engines
(this is the OpenRAG-vs-RAGFlow POC frame, extended with NeMo).

The answering LLM stays Gemini in all three deployments. The only
variable is retrieval/indexing — that is what is being evaluated.

## The engine surface Khoj needs

Everything the product layer uses today (from `app/main.py` and
`agent/main.py`), and how each engine provides it:

| Capability | OpenRAG (current) | RAGFlow | NeMo (this repo's adapter) |
|---|---|---|---|
| Ingest | `POST /router/upload_ingest`, poll `GET /tasks/{id}` | Upload to a dataset, then parse: `POST /api/v1/datasets/{id}/documents`, `POST /api/v1/datasets/{id}/chunks`; status from document `run`/`progress` | Adapter-side: extract text (pypdf/python-docx), chunk 1000/200, embed via NIM, upsert to Milvus |
| Search | `POST /search` with filename filters | `POST /api/v1/retrieval` with `dataset_ids` (+ optional `document_ids`), `similarity_threshold` | Embed query via NIM → Milvus search with scope filter → optional rerank NIM |
| Indexed check | `GET /files` | `GET /api/v1/datasets/{id}/documents` | Chunk count per `doc_id` in Milvus |
| Delete | `POST /documents/delete-by-filename` | `DELETE /api/v1/datasets/{id}/documents` | Milvus delete by `doc_id` expression |
| Per-doc filter (Knowledge tab) | knowledge-filter per document | retrieval with `document_ids=[...]` | Milvus filter `doc_id == ...` |

## Scoping

- OpenRAG: filename `{doc_id}--{filename}`, scope inside the name.
- RAGFlow: **one dataset per scope** (`users/{uid}`, `orgs/{orgId}`,
  plus the platform-org dataset as the global corpus). Retrieval
  passes the caller's dataset + the global dataset — structural
  isolation instead of name matching.
- NeMo: one Milvus collection per deployment; every chunk carries
  `scope` and `doc_id` fields and every query filters on them,
  mirroring the OpenRAG post-filter discipline.

## NeMo shape

NVIDIA NeMo Retriever is a set of NIM microservices (embedding,
reranking, ingestion) plus a vector store — the RAG blueprint runs
them on GPUs. Two ways to run it here:

1. **Hosted NIM APIs first** (integrate.api.nvidia.com): embedding
   `nvidia/llama-nemotron-embed-1b-v2`, rerank
   `nvidia/llama-nemotron-rerank-1b-v2`. Needs one NVIDIA API key
   (build.nvidia.com) in Secret Manager. No GPUs, deploys now.
2. **Self-hosted NIMs later**: GPU node pool + NGC API key + NIM
   Operator. The adapter does not change — only the base URLs do.

Chunking/extraction for NeMo happens in the adapter (the hosted path
has no ingestion service); that is an honest difference from the
other two engines and is recorded here rather than hidden.

## Deployments (GKE, cluster `khoj`)

- `khoj-openrag` — the existing stack, unchanged.
- `khoj-ragflow` — Khoj web+agent with `KHOJ_ENGINE=ragflow`, plus
  the RAGFlow stack (server, MySQL, Redis, MinIO, Elasticsearch).
  RAGFlow's own LLM/embedding config points at Gemini, set via its
  API after deploy.
- `khoj-nemo` — Khoj web+agent with `KHOJ_ENGINE=nemo`, plus Milvus
  standalone (with its etcd + MinIO). NVIDIA key via ExternalSecret.

Each product deployment gets its own ingress IP, its own Khoj data
volume (accounts/documents are per-deployment), and its own agent.

## Build order

1. `app/engines/` package: interface + OpenRAG implementation moved
   verbatim (no behaviour change; existing tests must pass as-is).
2. RAGFlow implementation + charts; deploy; verify register →
   upload → search → chat end to end.
3. NeMo implementation + charts; deploy; verify the same path.
4. Seed all three with the POC corpus + gold queries and bench.
