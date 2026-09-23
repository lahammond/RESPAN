# -*- coding: utf-8 -*-
"""CuPy / importlib-metadata compatibility shim.

CuPy enumerates installed distributions at import time. A malformed or partial
``.dist-info`` directory anywhere on the path - common in conda environments that
have been upgraded in place, and in the bundled environment this application
ships - yields a distribution whose metadata has no ``Name`` key, and CuPy raises
while reading it. This filters those out of ``distributions()`` so the import
succeeds.

Importing this module applies the patch. It is idempotent, it patches nothing it
cannot import, and it has no dependencies, so importing it is safe anywhere.

The packaged application applies the same patch through ``sitecustomize``, which
the interpreter imports at start-up. Applying both is harmless.
"""


def _patch(modname):
    try:
        m = __import__(modname, fromlist=['distributions'])
    except ImportError:
        return
    if getattr(m, "_cupy_fix", False):
        return
    orig = m.distributions

    def safe():
        for d in orig():
            if "Name" in d.metadata:
                yield d

    m.distributions, m._cupy_fix = safe, True


for _n in ("importlib_metadata", "importlib.metadata"):
    _patch(_n)
