#!/usr/bin/env python3
"""Unified entry point for the CLI, PySide UI, and FastAPI + web client."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import os
import sys
import threading
import time
import webbrowser
from http.client import HTTPConnection
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="T.O.M. entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the interactive terminal client (passes along additional args).",
    )
    parser.add_argument(
        "--pyside",
        action="store_true",
        help="Launch the PySide6 desktop shell.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("TOM_API_HOST", "127.0.0.1"),
        help="Host interface for the FastAPI server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("TOM_API_PORT", "8000")),
        help="Port for the FastAPI server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for API development.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically when starting the API + web UI.",
    )
    return parser


def _wait_for_server(host: str, port: int, timeout: float = 20.0) -> bool:
    """Poll until the HTTP server responds or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        conn = None
        try:
            conn = HTTPConnection(host, port, timeout=1.0)
            conn.request("GET", "/")
            resp = conn.getresponse()
            resp.read()
            if resp.status < 500:
                return True
        except OSError:
            time.sleep(0.3)
            continue
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        time.sleep(0.3)
    return False


def _open_browser_when_ready(host: str, port: int, path: str) -> None:
    """Poll the local server until it's reachable before opening the browser."""

    def _runner() -> None:
        target = f"http://{host}:{port}{path}"
        _wait_for_server(host, port)
        try:
            webbrowser.open(target)
        except Exception:
            pass

    threading.Thread(target=_runner, name="tom-browser-opener", daemon=True).start()


def _run_api_server(host: str, port: int, reload: bool, open_browser: bool) -> int:
    """Start the FastAPI app via uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime guard
        print("uvicorn is required to run the API server. Install fastapi[standard].", file=sys.stderr)
        raise SystemExit(1) from exc

    public_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    if open_browser:
        _open_browser_when_ready(public_host, port, "/web/")

    config = uvicorn.Config(
        "services.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
    if not reload:
        # Mute uvicorn access logs when running in embedded/CLI modes to avoid
        # duplicating HTTP traces in the terminal UI. When --reload is enabled
        # we keep the default logging so developers can see traffic clearly.
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    server = uvicorn.Server(config)
    try:
        return server.run()
    except KeyboardInterrupt:
        return 0
    except asyncio.CancelledError:
        return 0


def _run_cli(forwarded_args: Sequence[str]) -> None:
    """Invoke the CLI entry point, forwarding additional args."""
    from ui.cli.main import main as cli_main

    cli_main(list(forwarded_args))


def _has_cli_subcommand(args: Sequence[str], command: str) -> bool:
    return any(token == command for token in args)


def _has_flag(args: Sequence[str], flags: set[str]) -> bool:
    return any(token in flags for token in args)


def _has_arg(args: Sequence[str], flag: str) -> bool:
    for token in args:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def _run_cli_mode(host: str, port: int, reload: bool, forwarded_args: Sequence[str]) -> None:
    """Start the API if needed, then launch the CLI."""
    should_skip_server = (
        _has_cli_subcommand(forwarded_args, "clear-cache")
        or _has_arg(forwarded_args, "--api-base")
        or _has_flag(forwarded_args, {"-h", "--help"})
        or os.getenv("TOM_API_BASE")
    )

    server_proc: multiprocessing.Process | None = None
    public_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    if not should_skip_server:
        os.environ.setdefault("TOM_API_BASE", f"http://{public_host}:{port}")
        server_proc = multiprocessing.Process(
            target=_run_api_server,
            args=(host, port, reload, False),
            daemon=True,
        )
        server_proc.start()
        _wait_for_server(public_host, port)

    try:
        _run_cli(forwarded_args)
    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.join(timeout=5)


def _launch_pyside_ui() -> None:
    """Launch the PySide desktop UI."""
    try:
        from ui.pyside6.launcher import main as gui_main
    except ImportError as exc:  # pragma: no cover - PySide optional
        print("PySide6 is not installed. Install PySide6 to use the desktop UI.", file=sys.stderr)
        raise SystemExit(1) from exc

    gui_main()


def _run_pyside_mode(host: str, port: int, reload: bool) -> None:
    """Start the API server in the background, then launch the PySide UI."""
    poll_host = host if host not in {"0.0.0.0", "::"} else "127.0.0.1"
    server_proc = multiprocessing.Process(
        target=_run_api_server,
        args=(host, port, reload, False),
        daemon=True,
    )
    server_proc.start()
    _wait_for_server(poll_host, port)
    try:
        _launch_pyside_ui()
    finally:
        server_proc.terminate()
        server_proc.join(timeout=5)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args, remainder = parser.parse_known_args(argv)

    if args.cli and args.pyside:
        parser.error("Choose either --cli or --pyside, not both.")

    if args.cli:
        _run_cli_mode(args.host, args.port, args.reload, remainder)
        return
    if args.pyside:
        _run_pyside_mode(args.host, args.port, args.reload)
        return

    try:
        _run_api_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            open_browser=not args.no_browser,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
import logging
