# Private Workspace Knowledge protocol 1

The existing Partner AI service retains its shared PBR RAG knowledge and personal
chat behavior. This extension adds authenticated, stateless transcription and
embedding endpoints for Owner-reviewed Knowledge stored by Laravel.

- `POST /internal/pbr/private/extract`: inline PDF/JPEG/PNG/WebP base64 bytes;
  plain transcription plus a partial-extraction flag. No Files API or cache write.
- `POST /internal/pbr/private/embed`: separate typed Content items, one vector per
  text, explicit document/query prefixes and a model/dimension/protocol key. An
  aggregated or malformed response fails rather than indexing mismatched facts.
- `workspaceContext.scope=owner_private_knowledge_v1`: private citation rules;
  no Google Search/tools. Private errors and malformed bodies never echo input.
- Health advertises `privateKnowledgeProtocol: 1` so Laravel fails closed against
  an older service instead of silently using an incompatible private workflow.

No private source, vector, prompt or answer is added to shared chunks, caches or
PDFs. Original shared data and credentials are unchanged. The configured Gemini
provider still processes the content; this is application isolation, not a claim
about provider retention or server-operator access. Private source and chat
retention remain Laravel's explicit Owner-managed responsibility.

Run `node --test tests/private-knowledge.test.js` and the normal parse check.
No provider key or live data is needed for the contract tests. The paired Laravel
installer performs real synthetic-only OCR, embedding and cited generation checks
on the server before reopening real traffic.


## Chat interface protocol 1

Health also advertises `chatInterfaceProtocol: 1` and `replyLanguages`.
`POST /internal/pbr/chat` accepts `replyLanguage`: `auto` (default), `my`, or
`en`. Other values fail validation before streaming. Automatic replies follow
the latest question, so a conversation can switch between Burmese and English.
Explicit choices set the preferred reply language; an explicit translation or
language request in the latest question still wins. Replies are concise by
default, with fuller detail when asked. Source citations and necessary
uncertainty remain. No knowledge or access scope changes.

Run `node --test tests/*.test.js` for both private Knowledge and language contracts.

The current turn's reply-language rule is placed after all retrieved Knowledge and
Workspace evidence. Explicit English/Burmese choices apply even when the question,
history and sources use the other language; exact citations and business terms are
preserved. An explicit language request in the latest question still takes
precedence. This prevents the language of a long evidence block from becoming the
reply preference. Provider behavior is checked with synthetic English and Burmese
questions during the paired server release, in addition to the local contract tests.
