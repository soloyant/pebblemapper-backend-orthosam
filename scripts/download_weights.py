"""Fetch Segment Anything (SAM v1) checkpoints into weights/.

    python scripts/download_weights.py            # ViT-B (375 MB), the default
    python scripts/download_weights.py vit_l vit_h

The checkpoints are Meta's (Apache-2.0), the same files OrthoSAM's
orthosam-setup downloads. Set PM_ORTHOSAM_WEIGHTS to put them elsewhere.
None of them is redistributed by this repository.
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
import os
import sys
import urllib.request
from pathlib import Path

FILES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
BASE = "https://dl.fbaipublicfiles.com/segment_anything/"

wanted = sys.argv[1:] or ["vit_b"]
dest = Path(os.environ.get("PM_ORTHOSAM_WEIGHTS")
            or Path(__file__).resolve().parents[1] / "weights")
dest.mkdir(parents=True, exist_ok=True)
for key in wanted:
    if key not in FILES:
        sys.exit(f"unknown model {key!r}; choose from {', '.join(FILES)}")
    out = dest / FILES[key]
    if out.exists():
        print(f"have {out}")
        continue
    print(f"downloading {FILES[key]} ...", flush=True)
    urllib.request.urlretrieve(BASE + FILES[key], out)
    print(f"  -> {out} ({out.stat().st_size / 2**20:.0f} MB)")
sys.exit(0)
