"""OrthoSAM (Chan, Rheinwalt & Bookhagen, 2026) as a PebbleMapper detection backend.

Declared through ``user_detectors.json`` (``{"module":
"pm_orthosam_backend.backend", "factory": "make_backend", "path": "<this
repository>"}``); no PebbleMapper source is touched. The driver is
PebbleMapper's own :class:`detectors.instance_backend.InstanceSubprocessBackend`:
this file only says what the model is, where its weights are, and which
options its ``run.py`` takes.

``run.py`` runs in the ``pm-orthosam`` conda env: OrthoSAM's multi-scale
tiling around Segment Anything (SAM v1) leaves a label image, and every
instance is then measured by PebbleMapper's own measurement step, so the
table means exactly what Mask R-CNN's does. OrthoSAM tiles the image itself,
so in ortho mode each PebbleMapper job is handed over whole.
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
from pathlib import Path

from detectors.base import BackendInfo
from detectors.instance_backend import InstanceSubprocessBackend
from detectors import subprocess_runner

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
UPSTREAM = REPO_DIR / "upstream"
ENV_NAME = os.environ.get("PM_ORTHOSAM_ENV", "pm-orthosam")
_SCRIPT = PKG_DIR / "run.py"
WEIGHTS_DIR = Path(os.environ.get("PM_ORTHOSAM_WEIGHTS") or (REPO_DIR / "weights"))
CHECKPOINTS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
MODEL_TYPE = os.environ.get("PM_ORTHOSAM_MODEL", "vit_b")

CITATION = ("Chan, V., Rheinwalt, A., & Bookhagen, B. (2026). OrthoSAM: "
            "multi-scale extension of the Segment Anything Model for river "
            "pebble delineation from large orthophotos. Earth Surface "
            "Dynamics, 14, 391-416. https://doi.org/10.5194/esurf-14-391-2026")

INFO = BackendInfo(
    name="orthosam",
    display_name="OrthoSAM (Chan, Rheinwalt & Bookhagen)",
    framework="pytorch",
    license=("Apache-2.0 (OrthoSAM code, UP-RS-ESP); Apache-2.0 (Segment "
             "Anything code and checkpoints, Meta). None of it is "
             "redistributed with this adapter (MIT)."),
    output_type="instance",
    env=ENV_NAME,
    in_process=False,
    weights=str(WEIGHTS_DIR / CHECKPOINTS.get(MODEL_TYPE, CHECKPOINTS["vit_b"])),
    install_hint=("Run scripts/get_upstream.py (clones UP-RS-ESP/OrthoSAM into "
                  "upstream/), create the 'pm-orthosam' conda env "
                  "(scripts/create_env.sh), run scripts/download_weights.py and "
                  "declare the backend in user_detectors.json."),
    description=("Segment Anything (SAM v1) prompted on a point grid over "
                 "overlapping tiles, with coarser passes for grains larger than "
                 "a tile. Instances are measured by PebbleMapper's shared "
                 "measurement step; OrthoSAM reports no per-instance "
                 "confidence, so every clast is scored 1.0. Cite: " + CITATION),
)


def make_backend():
    """Zero-argument factory named in user_detectors.json."""
    return OrthoSamBackend()


class OrthoSamBackend(InstanceSubprocessBackend):
    info = INFO
    env_name = ENV_NAME
    required_modules = ("torch", "segment_anything", "cv2", "tifffile", "skimage")
    script = _SCRIPT
    model_version = "OrthoSAM 18da0e5 + SAM v1"
    log_prefix = "orthosam"

    def is_available(self) -> bool:
        ckpt = CHECKPOINTS.get(MODEL_TYPE, CHECKPOINTS["vit_b"])
        return (_SCRIPT.exists()
                and (UPSTREAM / "OrthoSAM" / "Layer_0.py").is_file()
                and (WEIGHTS_DIR / ckpt).exists()
                and subprocess_runner.conda_env_exists(ENV_NAME))

    def spec_params(self, mode, kwargs, resolution):
        log_fn = kwargs.get("log_fn") or (lambda s: None)
        if kwargs.get("min_confidence") is not None:
            log_fn("[orthosam] note: min_confidence is a Mask R-CNN threshold; "
                   "OrthoSAM has no detector confidence, so it is recorded in "
                   "the manifest but not applied.")
        opts = dict(kwargs.get("orthosam_options")
                    or json.loads(os.environ.get("PM_ORTHOSAM_OPTIONS", "{}") or "{}"))
        return {"weights_dir": str(WEIGHTS_DIR), "upstream": str(UPSTREAM),
                "model_type": MODEL_TYPE, "orthosam": opts}
