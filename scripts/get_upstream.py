"""Clone OrthoSAM (Apache-2.0) into upstream/, where run.py expects it.

    python scripts/get_upstream.py

Checks out the commit the adapter was tested with. OrthoSAM is used as is;
nothing of it is redistributed by this repository.
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
import subprocess
import sys
from pathlib import Path

URL = "https://github.com/UP-RS-ESP/OrthoSAM.git"
COMMIT = "18da0e57e6eb9aa7fd4085d4862c18597dae2723"

dest = Path(__file__).resolve().parents[1] / "upstream"
if (dest / "OrthoSAM" / "Layer_0.py").exists():
    print(f"have {dest}")
    sys.exit(0)
subprocess.run(["git", "clone", URL, str(dest)], check=True)
subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", COMMIT], check=True)
print(f"cloned into {dest} at {COMMIT[:7]}")
