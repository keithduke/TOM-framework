"""FastAPI entrypoint for T.O.M. core services."""

from fastapi import FastAPI

from .config import ApiSettings
from .runtime import ApiRuntime
from .routes import build_router

settings = ApiSettings.from_env()
runtime = ApiRuntime(settings)


app = FastAPI(title="T.O.M. API", version="0.2.0")
app.include_router(build_router(runtime))


@app.on_event("startup")
async def startup_event() -> None:
    runtime.startup()
