"""``python -m deepprep.synthesis`` — build a Synth-Spider/Synth-Bird set (Sec 5.3).

A dedicated entry point (rather than ``python -m deepprep.synthesis.synthesize``)
avoids re-executing a module that the package ``__init__`` has already imported.
"""

from __future__ import annotations

import sys

from .synthesize import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
