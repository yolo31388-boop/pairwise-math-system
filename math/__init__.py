"""Local ``math`` package.

This package shadows the standard-library ``math`` module.  Re-export the
stdlib module's public names so that code inside the package (and any
consumer that does ``import math`` after this package is loaded) still
finds ``sqrt``, ``sin`` and friends.
"""

import importlib.machinery as _importlib_machinery
import importlib.util as _importlib_util
import os as _os
import sysconfig as _sysconfig


def _load_stdlib_math():
    # math may be statically linked in (built-in) ...
    spec = _importlib_machinery.BuiltinImporter.find_spec("math")
    if spec is not None:
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    # ... or shipped as an extension module under lib-dynload.
    dynload = _os.path.join(_sysconfig.get_path("stdlib"), "lib-dynload")
    if _os.path.isdir(dynload):
        for name in sorted(_os.listdir(dynload)):
            stem, _, _ = name.partition(".")
            if stem == "math":
                spec = _importlib_util.spec_from_file_location(
                    "_stdlib_math", _os.path.join(dynload, name)
                )
                module = _importlib_util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    return None


_stdlib_math = _load_stdlib_math()
if _stdlib_math is not None:
    for _name in dir(_stdlib_math):
        if not _name.startswith("_"):
            globals()[_name] = getattr(_stdlib_math, _name)
    del _name
