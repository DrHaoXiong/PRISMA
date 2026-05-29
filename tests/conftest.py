"""Test configuration for repository-local imports.

GitHub Actions invokes pytest in a fresh Linux environment where the repository
root is not always placed on ``sys.path`` during test collection.  The public
PRISMA package is script-oriented rather than installed as a Python package, so
tests import modules such as ``builder`` and ``partition`` directly from the
repository root.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
