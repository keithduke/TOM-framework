# Sprint 1 — API-First Core

## Objective
Refactor T.O.M. into an API-first system so every interface (CLI, PySide6, web) talks to the same FastAPI backend. Finish Sprint 1 with a modular core that can be hosted as a service while keeping current clients functional.

## Architecture Blueprint
- `core/`: Model execution, context/history management, tool orchestration, caching, and configuration. No UI imports.
- `services/api/`: FastAPI app exposing session, messaging, tool, and admin endpoints using Pydantic schemas and dependency injection. Owns routers, background tasks, and startup/shutdown hooks.
- `ui/`: Presentation adapters.
  - `ui/cli/`: Terminal client that hits the API (initially may call core directly until endpoints stabilize).
  - `ui/pyside6/`: Desktop shell (Tray + window) that wraps the CLI or directly streams from the API.
  - `ui/web/`: Web UI served via FastAPI’s static mount or a separate frontend calling the same endpoints.

```
tom/
├─ core/
├─ services/
│  └─ api/
└─ ui/
   ├─ cli/
   ├─ pyside6/
   └─ web/
```

## Sprint 1 Deliverables
1. Extract existing LLM orchestration into `core/` modules with clean interfaces.
2. Stand up `services/api` skeleton (FastAPI app, health route, session/message contract).
3. Update CLI + PySide6 launchers to live under `ui/` and rely on the new package layout.
4. Document new commands and migration steps in `README`/`AGENTS`.
5. Keep regression tests (`pytest -q`) passing after the move.

## Current Status

- `core/` hosts all orchestration logic; CLI and PySide load it via `ui/`.
- `services/api` exposes health/session/chat endpoints (local-only, no auth) and serves the static web client.
- `python main.py` now boots the FastAPI server and opens the browser, so the web UI is the default experience.
- Running `python main.py --cli` automatically starts the local API (unless `--api-base` is provided) and points the CLI at it.
- PySide6 launcher (`--pyside` or legacy `launcher.py`) likewise spins up/attaches to the API and mirrors tool/thinking output.

## Next Steps

- Implement streaming (`/chat/stream` SSE) so CLI, PySide, and web share a consistent real-time contract.
- Continue evolving `ui/web/` (session list, cache stats, prompt/tool inspectors) to reach CLI feature parity.
- Add lightweight ops polish (metrics, structured logs, optional auth when binding beyond localhost).
## UI Launch Strategy

- `python main.py` (default) spins up the local FastAPI server + serves the web UI. Everything stays on-device, but the browser is now the primary entry point.
- `python main.py --cli` keeps the classic terminal workflow while defaulting to the local API (set `--api-base` or `TOM_API_BASE` to target a remote backend, or call `ui/cli/main.py` directly for legacy direct-core mode).
- `python main.py --pyside` launches the desktop shell and proxies through the API; legacy `launcher.py` remains for shortcuts/scripts.
- All three adapters live under `ui/` and remain thin; `core/` plus `services/api/` own the business logic, tool execution, and cache management.
