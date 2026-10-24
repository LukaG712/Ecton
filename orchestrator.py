"""Junie Orchestrator — backward-compatible entry point.

This module is a thin wrapper around the `junie_adapter` package.
Use `python -m orchestrator` or `uvicorn orchestrator:app` to start the HTTP service.

For direct module usage, import from `junie_adapter` instead:
    from junie_adapter import run_junie_task, OrchestratorConfig
"""

from junie_adapter.service import app  # noqa: F401

if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("junie_adapter.service:app", host=host, port=port, reload=False)
