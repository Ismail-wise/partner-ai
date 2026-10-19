# PBR AI Integration Service

This service is a private server-to-server adapter for the PBR Laravel platform.

## Important safety rule

The original `server.js`, `index.html`, `admin.html`, `docs/` folder and `vectordb_cache.json` remain the standalone Partner AI application and knowledge base. The PBR service does not delete or rewrite them.

The original cache contains the friend's preserved chunk text/source plus embeddings created by the older embedding model. PBR keeps that cache untouched, reuses its exact chunk text/source when the PDF fingerprint matches, and creates a separate PBR vector cache using the current PBR embedding model. This protects the original knowledge work while allowing PBR retrieval to use a supported embedding space.

If the legacy cache cannot provide the original chunks, PBR falls back to extracting the same `docs/` PDFs with the existing 500-word / 100-word-overlap chunking approach.

## Required environment variables

- `GEMINI_API_KEY` — Gemini API key used by Partner AI.
- `PBR_INTERNAL_SECRET` — long random secret shared only between Laravel and this localhost service. Minimum 32 characters.

## Optional environment variables

- `PBR_INTERNAL_HOST=127.0.0.1`
- `PBR_INTERNAL_PORT=3107`
- `PBR_AI_MODEL=gemini-2.5-flash`
- `PBR_EMBEDDING_MODEL=gemini-embedding-2`
- `PBR_EMBEDDING_DIMENSIONS=768`
- `PBR_EMBEDDING_BATCH_SIZE=16`
- `PBR_VECTOR_CACHE_PATH=/var/lib/pbr-ai/pbr_vectordb_cache.json`
- `PBR_DOC_THRESHOLD=0.45`
- `PBR_MAX_HISTORY=20`
- `PBR_MAX_CONTEXT_CHARS=60000`
- `PBR_MAX_MESSAGE_CHARS=5000`

The embedding cache builder retries temporary 429 / 503 / resource-exhausted failures with bounded backoff. A failed batch is also retried chunk-by-chunk so one bad item does not discard the whole batch.

## Install

The standalone chatbot still uses its existing `@google/generative-ai` dependency. PBR additionally uses the current `@google/genai` SDK, so both are intentionally present.

```bash
npm install --package-lock=false
```

## Commands

Syntax check:

```bash
npm run check:pbr
```

Start private PBR service:

```bash
npm run start:pbr
```

Health check from the same server:

```bash
curl -s http://127.0.0.1:3107/health
```

On the first start, PBR may need to create its separate vector cache. Later starts reuse that cache as long as the docs, embedding model and embedding dimensions have not changed. For production, `PBR_VECTOR_CACHE_PATH` should point to a writable runtime directory such as `/var/lib/pbr-ai/`, allowing the Git checkout itself to remain read-only to the service user.

## Private API

`POST /internal/pbr/chat`

Required header:

```text
X-PBR-Internal-Secret: <shared secret>
```

The route accepts the user message, recent PBR chat history and an already-authorized workspace context snapshot from Laravel. It returns Server-Sent Events (SSE): `meta`, `delta`, `error`, and `done` events.

The service binds to `127.0.0.1` by default and should not be exposed directly through Nginx or a public browser endpoint.
