# Deploying the RAGFlow and NeMo stacks in GKE

Two more Khoj deployments next to the live OpenRAG one, same
cluster (`khoj`), one namespace each — the charts hardcode service
names, so namespaces are the isolation boundary:

| Stack | Namespace | Engine services | Khoj engine |
|---|---|---|---|
| OpenRAG (live) | `default` | existing openrag-* releases | `openrag` |
| RAGFlow | `khoj-ragflow` | charts/ragflow (this repo) | `ragflow` |
| NeMo | `khoj-nemo` | Milvus official chart (standalone) | `nemo` |

Shared GSM secrets are reused across namespaces (ESO syncs per
namespace): `khoj-auth-secret`, `khoj-jira-api-token`,
`openrag-openai-api-key` (the Gemini key). New ones below.

## 0. Images

Build the web + agent images from this repo into the same Artifact
Registry repo as the OpenRAG stack, with a distinct tag:

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/innovation-lab-2026/khoj/khoj:multi-0.1.0 .
gcloud builds submit --tag us-central1-docker.pkg.dev/innovation-lab-2026/khoj/khoj-agent:multi-0.1.0 -f agent/Dockerfile .
```
(Use the repo's Dockerfiles; the deploys below pass
`--set image.tag=multi-0.1.0`.)

## 1. RAGFlow stack (namespace khoj-ragflow)

1. Create the engine passwords in GSM (generate in-shell, never print):
   `ragflow-mysql-password`, `ragflow-elastic-password`,
   `ragflow-minio-password` — e.g.
   `openssl rand -base64 24 | tr -d '\n' | gcloud secrets create ragflow-mysql-password --data-file=-`
2. `kubectl create namespace khoj-ragflow`
3. Install the engine:
   `helm install ragflow charts/ragflow -n khoj-ragflow`
   (docEngine defaults to infinity — as deployed 2026-11-06:
   Autopilot's Warden webhook rejects the vm.max_map_count sysctl
   the Elasticsearch pod needs, so ES is not installable on this
   cluster. The comparison notes this stack runs Infinity.)
4. Wait for all pods Running; open the RAGFlow UI
   (`kubectl -n khoj-ragflow port-forward svc/ragflow 8080:80` or
   enable `server.ingress`), register the admin account, then in
   RAGFlow's model settings add Google Gemini as chat + embedding
   provider (the same Gemini key as the rest of the stack).
5. Create a RAGFlow API key (avatar menu → API → create key) and
   store it in GSM as `ragflow-api-key`.
6. Install Khoj into the same namespace:
   `helm install khoj charts/khoj -n khoj-ragflow --set image.tag=multi-0.1.0 --set engine.name=ragflow --set engine.ragflowExternalSecret.enabled=true --set agent.url=http://khoj-agent:8080`
   and the agent:
   `helm install khoj-agent charts/khoj-agent -n khoj-ragflow --set image.tag=multi-0.1.0 --set engine.name=ragflow`
   (Do NOT set engine.ragflowExternalSecret.enabled=true on the agent: the web chart owns the shared ExternalSecret `khoj-ragflow`, and a second owner fails helm's ownership check — live finding 2026-11-07. The agent reads the same synced Secret by name. Same rule for the NeMo engine's nvidia ExternalSecret.)
   (agent `web.url` must point at the in-namespace web service —
   check values before installing.)
7. Give the Khoj release its own ingress/IP (same pattern as the
   OpenRAG stack) and verify: register → login → upload → search →
   chat → delete, plus an unrelated query returning zero results.

## 2. NeMo stack (namespace khoj-nemo)

1. Store the NVIDIA API key (build.nvidia.com — free NGC key) in
   GSM as `nvidia-api-key`.
2. `kubectl create namespace khoj-nemo`
3. Install Milvus standalone from the official chart:
   `helm repo add milvus https://zilliztech.github.io/milvus-helm/`
   `helm install milvus milvus/milvus -n khoj-nemo -f charts/milvus/values-standalone.yaml`
   The service is `milvus:19530` in-namespace, matching the chart
   default `engine.milvusUrl`.
4. Install Khoj + agent with `--set engine.name=nemo --set engine.nvidiaExternalSecret.enabled=true`
   (same shape as step 6 above).
5. Verify the same end-to-end path. First upload creates the
   `khoj_chunks` collection; if the embedding dimension differs
   from 1024 the engine error names it — set `NEMO_EMBED_DIM` via
   chart env (add when needed) and retry.

## Notes

- HTTPS: these stacks inherit the same open HTTPS decision as the
  OpenRAG stack (domain + managed certs vs sslip.io + Let's
  Encrypt). Whatever route is chosen there applies here unchanged.
- Bench: once all three are live, seed the same corpus
  (rag-pocs sample docs) into each and run the 12 gold queries per
  engine; scores are NOT comparable across engines (each engine's
  relevance floor differs — OpenRAG 1.5, RAGFlow similarity,
  NeMo cosine 0.35 starting point) — compare answer quality and
  latency, not raw scores.
