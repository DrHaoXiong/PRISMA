#!/usr/bin/env python3
"""Import-check the core PRISMA runtime dependencies."""

from __future__ import annotations

import importlib
import sys


REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("polars", "polars"),
    ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"),
]

OPTIONAL_PACKAGES = [
    ("bed_reader", "bed-reader"),
]


def _check(import_name: str, display_name: str, required: bool) -> bool:
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "version unavailable")
        print(f"[OK] {display_name}: {version}")
        return True
    except Exception as exc:
        level = "ERROR" if required else "WARNING"
        print(f"[{level}] {display_name}: import failed ({exc})")
        return not required


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    ok = True
    for import_name, display_name in REQUIRED_PACKAGES:
        ok = _check(import_name, display_name, required=True) and ok
    for import_name, display_name in OPTIONAL_PACKAGES:
        ok = _check(import_name, display_name, required=False) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
