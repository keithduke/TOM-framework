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
