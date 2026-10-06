"""Pytest bootstrap for the repository.

Two environment issues are worked around here so the shipped tests can run:

1. The local ``math`` package shadows the standard-library ``math``
   module.  Pytest imports stdlib ``math`` during startup, so by the time
   ``tests/test_math.py`` runs ``from math.system import MathSystem`` the
   name ``math`` is already bound to the stdlib module (which is not a
   package) and the import fails.  We register the local package in
   ``sys.modules`` before collection, re-exporting the stdlib names so
   nothing else breaks.

2. The test classes are plain classes (not ``unittest.TestCase``) but use
   ``self.assertIsNotNone``.  A minimal assertion helper is injected at
   collection time.
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _register_local_math_package():
    stdlib_math = sys.modules.get("math")
    package_dir = _ROOT / "math"
    spec = importlib.util.spec_from_file_location(
        "math",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["math"] = module
    spec.loader.exec_module(module)
    if stdlib_math is not None:
        for name in dir(stdlib_math):
            if not name.startswith("_") and not hasattr(module, name):
                setattr(module, name, getattr(stdlib_math, name))


_register_local_math_package()


def _assert_is_not_none(self, obj, msg=None):
    if obj is None:
        raise AssertionError(msg or "expected a non-None value, got None")


def pytest_collection_modifyitems(items):
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is not None and not hasattr(cls, "assertIsNotNone"):
            cls.assertIsNotNone = _assert_is_not_none
