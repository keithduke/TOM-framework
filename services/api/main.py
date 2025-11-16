"""FastAPI entrypoint for T.O.M. core services."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ApiSettings
from .runtime import ApiRuntime
from .routes import build_router

settings = ApiSettings.from_env()
runtime = ApiRuntime(settings)
WEB_STATIC_DIR = Path(__file__).resolve().parents[2] / "ui" / "web" / "static"
WEB_INDEX = WEB_STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.startup()
    try:
        yield
    finally:
        runtime.shutdown()


app = FastAPI(title="T.O.M. API", version="0.2.0", lifespan=lifespan)
app.include_router(build_router(runtime))


def _api_status() -> dict[str, str]:
    return {"message": "T.O.M. API is running"}


if WEB_STATIC_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_STATIC_DIR, html=True), name="web")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> Response:
        if WEB_INDEX.exists():
            return FileResponse(WEB_INDEX)
        return JSONResponse(_api_status())

else:

    @app.get("/", include_in_schema=False)
    async def root_status() -> Response:
        return JSONResponse(_api_status())
