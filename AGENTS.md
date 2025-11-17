# Repository Guidelines

## Project Structure & Module Organization
The shared engine now lives in `core/` (`config.py`, `context_manager.py`, `model_manager.py`, `prompt_cache_manager.py`, `tools.py`, `utils.py`). UI adapters sit under `ui/`: `ui/cli/` holds the terminal entry point plus interface, `ui/pyside6/` hosts the desktop launcher, and `ui/web/` serves the browser client. `services/api/` owns the FastAPI app, routers, schemas, SSE streaming loop, and DI wiring (see `SPRINT1.md`). Root-level `main.py` is the unified entry point for every adapter, while `launcher.py` simply shims to PySide mode for backward compatibility. Tests such as `test_tool_system.py`, `test_prompt_building.py`, `test_end_to_end.py`, and `test_api_endpoints.py` in the repo root still cover tool execution, prompt wiring, cache behavior, and API streaming. MLX-converted weights are expected under directories like `Qwen3-4B-Thinking-2507-8bit`, while Prompt Toolkit history and cache files continue to live next to the binaries.

## Build, Test, and Development Commands
```
python3.11 -m venv venv && source venv/bin/activate   # create & activate env
pip install -r requirements.txt                       # install MLX, FastAPI, PySide6, etc.
python main.py                                        # start FastAPI + web UI (default)
python main.py --cli [CLI flags]                      # run the terminal adapter via API
python main.py --pyside                               # launch the PySide6 desktop shell
python launcher.py                                    # legacy shortcut -> PySide mode
pytest -q                                             # run all tests; add -k name to target
```
Prefer passing `--debug`, `--gc-frequency`, or `--no-cache` to `main.py --cli` when validating context handling edge cases. When embedding the runtime elsewhere, export `TOKENIZERS_PARALLELISM=true` before forking to avoid the Hugging Face tokenizer warning (the standard entry points already do this).

## Coding Style & Naming Conventions
Use Python 3.11, four-space indentation, and descriptive class/function names that mirror their modules (e.g., `ContextManager`, `ModelManager`). Keep public helpers documented with concise docstrings, log via `logging` + `rich.logging.RichHandler`, and guard entry points with `if __name__ == "__main__"`. Tool definitions follow snake_case keys, while model/config paths use kebab-free directory names such as `Qwen3-0.6B-4bit`.

## Testing Guidelines
Tests run under `pytest`; each file is named `test_<area>.py` with functions prefixed `test_`. Add integration coverage for new tool flows (extend `test_end_to_end.py`) and unit coverage for prompt math or caching behavior. Keep fixtures fast—mock MLX heavy calls when possible and gate filesystem writes under `/tmp`. Validate that `context_manager` bounds, tool extraction, and cache trimming survive repeated runs before opening a PR. For manual verification, run `python main.py`, `python main.py --cli`, and `python main.py --pyside` to ensure all adapters stream thinking/tool events exactly once.

## Commit & Pull Request Guidelines
Recent history favors short, action-focused commit subjects (e.g., "Fix tool calling loop" or "Ignore history"), so keep messages under ~60 chars with imperative voice and follow up with optional body text for context. PRs should describe the scenario, include CLI/launcher repro steps, mention any model/cache migrations, and attach logs or screenshots when UI behavior changes. Link GitHub issues, check `pytest -q`, and confirm `python main.py` still boots with default options before requesting review.

## Security & Configuration Tips
Avoid committing converted weights or `.tom_history`; gitignore already excludes generated artifacts—double-check before pushing. Favor environment variables or `config.py` for secrets instead of embedding values in prompts. When sharing cache files, scrub user transcript data and respect OS sandbox permissions.
