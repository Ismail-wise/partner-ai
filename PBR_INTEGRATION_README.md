# PBR AI Integration Service

This service is a private server-to-server adapter for the PBR Laravel platform.

## Important safety rule

The original `server.js`, `index.html`, `admin.html`, `docs/` folder and `vectordb_cache.json` remain the standalone Partner AI application and knowledge base. The PBR service does not delete, rewrite or require replacing them.

`pbr-service.js` first attempts to reuse the existing `vectordb_cache.json` when its PDF fingerprint still matches the current `docs/` folder. If that legacy cache is unavailable or stale, it creates a separate ignored file named `pbr_vectordb_cache.json`.

## Required environment variables

- `GEMINI_API_KEY` — Gemini API key used by the existing knowledge engine.
- `PBR_INTERNAL_SECRET` — long random secret shared only between Laravel and this localhost service. Minimum 32 characters.

## Optional environment variables

- `PBR_INTERNAL_HOST=127.0.0.1`
- `PBR_INTERNAL_PORT=3107`
- `PBR_AI_MODEL=gemini-2.5-flash`
- `PBR_EMBEDDING_MODEL=text-embedding-004`
- `PBR_DOC_THRESHOLD=0.45`
- `PBR_MAX_HISTORY=20`
- `PBR_MAX_CONTEXT_CHARS=60000`
- `PBR_MAX_MESSAGE_CHARS=5000`

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

## Private API

`POST /internal/pbr/chat`

Required header:

```text
X-PBR-Internal-Secret: <shared secret>
```

The route accepts the user message, recent PBR chat history and an already-authorized workspace context snapshot from Laravel. It returns Server-Sent Events (SSE): `meta`, `delta`, `error`, and `done` events.

The service is intended to bind to `127.0.0.1` and should not be exposed directly through Nginx or a public browser endpoint.
