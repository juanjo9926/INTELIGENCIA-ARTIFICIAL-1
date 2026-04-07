"""Carga variables desde `.env` en la raiz del proyecto y expone rutas a datos."""
from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def load_env() -> Path:
    root = get_project_root()
    env_file = root / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, encoding="utf-8")
    except ImportError:
        pass
    # Si .env deja PROJECT_ROOT vacio, ignorarlo (Path('') seria cwd y rompe rutas)
    if not (os.environ.get("PROJECT_ROOT") or "").strip():
        os.environ["PROJECT_ROOT"] = str(root)
    return root


def data_path(var_name: str) -> Path:
    """Ruta absoluta al fichero definido por `var_name` en `.env`."""
    load_env()
    raw = (os.environ.get("PROJECT_ROOT") or "").strip()
    root = Path(raw) if raw else get_project_root()
    rel = os.environ.get(var_name)
    if not rel:
        raise KeyError(f"Variable de entorno {var_name!r} no definida en .env")
    return (root / rel).resolve()
