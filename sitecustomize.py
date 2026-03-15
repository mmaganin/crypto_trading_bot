from __future__ import annotations

import site
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
USER_SITE = Path(site.getusersitepackages())

for dependency_dir in (USER_SITE, ROOT / ".vendor", ROOT / ".deps"):
    has_numpy = (dependency_dir / "numpy" / "__init__.py").exists()
    has_pandas = (dependency_dir / "pandas" / "__init__.py").exists()
    if dependency_dir.exists() and has_numpy and has_pandas:
        sys.path.insert(0, str(dependency_dir))
