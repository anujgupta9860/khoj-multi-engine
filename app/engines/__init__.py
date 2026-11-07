"""Engine selection: KHOJ_ENGINE=openrag|ragflow|nemo (default
openrag). One engine per deployment — see docs/multi-engine-plan.md.
"""

from __future__ import annotations

import os

from .base import Engine, EngineError  # noqa: F401  (re-exported)
from .openrag import OpenRAGEngine

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    name = os.environ.get("KHOJ_ENGINE", "openrag").strip().lower()
    if name == "openrag":
        _engine = OpenRAGEngine()
    elif name == "ragflow":
        from .ragflow import RAGFlowEngine

        _engine = RAGFlowEngine()
    elif name == "nemo":
        from .nemo import NeMoEngine

        _engine = NeMoEngine()
    else:
        raise EngineError(f"Unknown KHOJ_ENGINE: {name!r}")
    return _engine
