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
- `services/api` exposes health/session/chat endpoints with API-key auth + telemetry.
- CLI can run in local mode (default) or proxy through the API via `--api-base/--api-key`.
- PySide6 launcher now talks directly to the API and mirrors tool/thinking output.
- `python main.py` still launches the CLI for now; `--cli`/`--pyside` flags keep explicit modes until the web UI is ready.

## Next Steps

- Make `python main.py` start the FastAPI server + serve the upcoming web client by default.
- Implement `ui/web/` (vanilla HTML/CSS/JS) on top of the existing API contract.
- Add lightweight ops polish (metrics, structured logs, auth UX) once all adapters share the backend.
## UI Launch Strategy

- `python main.py` (default) will evolve into the local FastAPI server + web UI launcher once Phase 3 lands. It keeps everything local-only but provides a single entry point for the browser experience.
- `python main.py --cli` keeps the classic terminal workflow. Today it can run in direct-core mode; once the API is fully hardened it will default to calling the local API (`--api-base` flag) so the CLI, PySide, and web clients all share the same runtime.
- `python main.py --pyside` launches the desktop shell. Like the CLI, it will transition to API-backed calls so every UI is just an adapter over the shared engine.
- All three adapters live under `ui/` and remain thin; `core/` plus `services/api/` own the business logic, tool execution, and cache management.
