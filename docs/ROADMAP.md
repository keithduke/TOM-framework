# T.O.M. Roadmap

This roadmap captures the near-term milestones for evolving T.O.M. from a CLI-first assistant into a multi-interface, API-driven platform. Each phase builds on the refactor completed in **Sprint 1**.

## Phase 1 — API-First Core (✅)
- Extracted orchestration into `core/`.
- Introduced `services/api` FastAPI skeleton with health/session routes.
- Moved CLI/PySide6 adapters under `ui/` with compatibility shims.
- Updated docs/tests to reflect the new layout.

## Phase 2 — Service Hardening (🚧)
- Flesh out `services/api` with session lifecycle, message streaming (SSE first, WebSockets later), and tool execution endpoints.
- Wire the CLI to call the API (toggle between direct-core and HTTP mode) so every adapter rides the same backend.
- Keep auth optional: default local-only mode, with env-driven API key if someone binds beyond localhost.
- Expand tests (`test_end_to_end.py`, `test_api_endpoints.py`) to cover HTTP flows via `httpx`.

## Phase 3 — Web Client (🚧)
- Finish `ui/web/` features: better session controls, streaming output, cache stats, and devtools for prompt/tool inspection.
- Share the session timeline across CLI, PySide6, and web; each adapter should be able to reconnect to an existing session.
- Layer a minimal design system (thinking/tool states, error handling) so the browser experience matches CLI clarity.

## Phase 4 — Deployment & Ops
- Containerize the FastAPI service (uvicorn worker sizing, MLX config).
- Provide deployment guides (local, Fly.io/FastAPI Cloud, etc.).
- Add observability: structured logging, request metrics, cache hit telemetry.

## Phase 5 — Advanced Features
- Tool marketplace/registry (dynamic loading, permissions).
- Multi-model orchestration (routing, fallbacks).
- Long-context enhancements (disk-backed history, smarter trimming).
- Fine-grained access control and audit logging for enterprise deployments.

_Last updated: 2025-11-16_
