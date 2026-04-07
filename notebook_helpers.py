# -*- coding: utf-8 -*-
"""Utilidades para localizar la raiz del proyecto y guardar notebooks."""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Devuelve la carpeta primerparcial_ia (donde esta este archivo o load_project_env)."""
    here = Path(__file__).resolve().parent
    return here


def find_data_path(*parts: str) -> Path:
    """Construye ruta absoluta bajo la raiz del proyecto."""
    return project_root().joinpath(*parts)
