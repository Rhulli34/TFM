"""Environment-aware configuration for the Financial News Radar API.

The frontend is served as static files from the same FastAPI process, so there
is no separate frontend origin to configure.  CORS is open in both modes
because all browser requests are same-origin (port 8000).

Key environment variables
-------------------------
APP_ENV   "dev" | "prod"   (default: "dev")
DB_PATH   Override the SQLite path.  In prod the image ships a baked-in snapshot
          of data/processed/radar.db; docker-compose.yml mounts a fresher copy on
          top for local dev.  Set DB_PATH to use a custom path in either mode.
PORT      Listen port in prod (default: 8000).  Also used by docker-compose.

Modes
-----
dev   Hot-reload on, binds to 127.0.0.1.  Set in local .env or omit APP_ENV.
prod  No reload, binds to 0.0.0.0 (required inside a container).
      Set APP_ENV=prod in docker-compose.yml environment section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.data.store import DB_PATH as _DEFAULT_DB


@dataclass
class Settings:
    env: str
    db_path: Path
    cors_origins: list[str]
    host: str
    port: int
    debug: bool


def get_settings() -> Settings:
    """Build a Settings instance from the current environment."""
    env = os.getenv("APP_ENV", "dev").lower()
    db_path = Path(os.getenv("DB_PATH", str(_DEFAULT_DB)))

    if env == "prod":
        return Settings(
            env="prod",
            db_path=db_path,
            cors_origins=["*"],   # frontend is same-origin; CORS headers are for external callers
            host="0.0.0.0",       # bind to all interfaces inside the container
            port=int(os.getenv("PORT", "8000")),
            debug=False,
        )

    return Settings(
        env="dev",
        db_path=db_path,
        cors_origins=["*"],
        host="127.0.0.1",
        port=8000,
        debug=True,
    )
