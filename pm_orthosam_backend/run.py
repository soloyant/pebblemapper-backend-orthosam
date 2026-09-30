"""OrthoSAM backend: subprocess entry point (runs in conda env 'pm-orthosam').

Reads the PebbleMapper job spec (``--spec <json>``), runs OrthoSAM's layers on
each job image and writes, for each job, an *instance file* next to the CSV
the core expects::

    <out_csv>.instances.npz   labels  int32 HxW  (0 = background, k = grain k)
                              scores  float32 N  (all 1.0: OrthoSAM reports no
                                                  per-instance confidence)
                              shape   (H, W)

It deliberately does NOT write the canonical CSV: measurement happens in the
core (detectors.measure.measure_mask) so the numbers are defined identically
to Mask R-CNN. This file must never import the PebbleMapper core.

OrthoSAM is used unmodified from ``upstream/`` (a clone of UP-RS-ESP/OrthoSAM).
Its package ``__init__`` imports the synthetic-data tools (numba, pycocotools),
which detection does not need, so the package is registered without running
it and only the three pipeline modules are imported. Each job runs in a
scratch folder shaped the way OrthoSAM expects (DataDIR/<dataset>/<file>,
OutDIR/chunks, OutDIR/Merged), which is deleted afterwards.

Spec params honoured: ``devicemode`` ("gpu"/"cpu"), ``weights_dir``,
``upstream``, ``model_type`` (vit_b/vit_l/vit_h), ``orthosam`` (dict):

    upsample          resample factor of the first pass (default 1). OrthoSAM
                      finds grains reliably from about 30 px across; 2 or 3
                      brings finer grains above that, at 4 to 9 times the cost.
    coarse_passes     resample factors of the later passes, relative to the
                      original image (default [upsample / 2]); [] for none.
    tile_size         SAM tile in pixels (default 1024)
    tile_overlap      overlap between tiles in pixels (default 200)
    points_per_axis   SAM prompt grid per tile (default 30); GPU memory grows
                      with its square
    stability_t       SAM stability-score threshold (default 0.85)
    dilation_size     OrthoSAM mask dilation (default 5)
    min_area_px       smallest grain kept, in original pixels (default 30)
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")   # OrthoSAM calls plt.show()

CHECKPOINTS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}


def log(msg):
    print(f"[orthosam] {msg}", flush=True)


def _load_rgb(path, mode):
    """HxWx3 uint8 in the stored pixel frame (no EXIF transpose, like the core).
    Ortho GeoTIFFs are read with rasterio (first three bands, scaled to 8 bit)."""
    if mode == "ortho":
        import rasterio
        with rasterio.open(path) as ds:
            n = min(3, ds.count)
            arr = ds.read(list(range(1, n + 1)))
            arr = np.moveaxis(arr, 0, -1)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            lo, hi = np.nanpercentile(a, 0.5), np.nanpercentile(a, 99.5)
            a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255
            arr = a.astype(np.uint8)
        return np.ascontiguousarray(arr)
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _import_orthosam(upstream):
    """Import OrthoSAM's pipeline modules without running its package __init__."""
    pkg_dir = os.path.join(upstream, "OrthoSAM")
    if not os.path.isfile(os.path.join(pkg_dir, "Layer_0.py")):
        raise FileNotFoundError(f"OrthoSAM not found in {upstream}; "
                                "run scripts/get_upstream.py")
    if "OrthoSAM" not in sys.modules:
        pkg = types.ModuleType("OrthoSAM")
        pkg.__path__ = [pkg_dir]
        sys.modules["OrthoSAM"] = pkg
    from OrthoSAM.Layer_0 import predict_tiles
    from OrthoSAM.Merging import merge_chunks
    from OrthoSAM.Layer_n import predict_tiles_n
    return predict_tiles, merge_chunks, predict_tiles_n


def _para_list(opts, data_dir, out_dir, ckpt_dir, model_type):
    """The per-pass parameter dicts OrthoSAM's utility.setup() would build.

    Sizes are kept in pixels: resolution(mm) is 1, so expected_min_size(sqmm)
    is an area in original pixels (OrthoSAM scales it by each pass's factor).
    """
    up = float(opts.get("upsample", 1))
    passes = opts.get("coarse_passes")
    if passes is None:
        passes = [up / 2]
    master = {
        "MODEL_TYPE": model_type,
        "CheckpointDIR": ckpt_dir,
        "DataDIR": data_dir,
        "MainOutDIR": out_dir,
        "BaseDIR": out_dir,
        "OutDIR": out_dir,
        "DatasetName": "job",
        "fid": "image.npy",
        "resolution(mm)": 1,
        "tile_size": int(opts.get("tile_size", 1024)),
        "tile_overlap": int(opts.get("tile_overlap", 200)),
        "resample_factor": up,
        "1st_resample_factor": up,
        "input_point_per_axis": int(opts.get("points_per_axis", 30)),
        "dilation_size": int(opts.get("dilation_size", 5)),
        "stability_t": float(opts.get("stability_t", 0.85)),
        "expected_min_size(sqmm)": float(opts.get("min_area_px", 30)),
        "min_radius": 0,
        "edge_removal": True,
        "Calculate_stats": False,
        "Discord_notification": False,
        "Plotting": False,
    }
    lst = [dict(master)] + [dict(master, resample_factor=float(f)) for f in passes]
    with open(os.path.join(out_dir, "para.json"), "w") as fh:
        json.dump(lst, fh, indent=2)
    with open(os.path.join(out_dir, "pre_para.json"), "w") as fh:
        json.dump([{} for _ in lst], fh)
    return lst


def _guard_saves():
    """Make every numpy save recreate its folder first.

    OrthoSAM creates each pass's chunk folder once, then fills it tile by tile.
    On Windows an empty folder under the temporary directory was twice removed
    between its creation and the first tile's save, several minutes later, which
    killed the run. This process only runs OrthoSAM, so wrapping numpy.save here
    touches nothing else.
    """
    if getattr(np.save, "_pm_guarded", False):
        return
    original = np.save

    def save(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            os.makedirs(os.path.dirname(os.fspath(file)) or ".", exist_ok=True)
        return original(file, *args, **kwargs)

    save._pm_guarded = True
    np.save = save


def _segment(image, opts, ckpt_dir, model_type, upstream, work_root):
    """Run OrthoSAM on one RGB array; int32 label image at the input's size."""
    import cv2
    from skimage.segmentation import relabel_sequential

    predict_tiles, merge_chunks, predict_tiles_n = _import_orthosam(upstream)
    _guard_saves()
    os.makedirs(work_root, exist_ok=True)
    work = tempfile.mkdtemp(prefix="pmos_", dir=work_root)
    try:
        data_dir = os.path.join(work, "data")
        out_dir = os.path.join(work, "out")
        os.makedirs(os.path.join(data_dir, "job"))
        for sub in ("", "chunks", "Merged"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        np.save(os.path.join(data_dir, "job", "image.npy"), image)
        paras = _para_list(opts, data_dir, out_dir, ckpt_dir, model_type)
        for n in range(len(paras)):
            t0 = time.time()
            if n == 0:
                predict_tiles(paras, 0)
                merge_chunks(paras, 0)
            else:
                predict_tiles_n(paras, n)
            log(f"  pass {n} (resample {paras[n]['resample_factor']}) "
                f"{time.time() - t0:.1f}s")
        merged = sorted(glob.glob(os.path.join(out_dir, "Merged", "Merged_Layers_*.npy")))
        if not merged:
            return np.zeros(image.shape[:2], dtype=np.int32)
        # A later pass that finds nothing new saves no file: the last one saved
        # holds every pass so far. Its grid is the first pass's (resampled).
        lab = np.asarray(np.load(merged[-1], allow_pickle=True))
        if lab.shape[:2] != image.shape[:2]:
            lab = cv2.resize(lab.astype(np.float32), (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        lab = relabel_sequential(lab.astype(np.int64))[0]
        return lab.astype(np.int32)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="OrthoSAM backend for PebbleMapper.")
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    mode = str(spec.get("mode", "quadrat")).lower()
    params = spec.get("params", {})
    opts = params.get("orthosam", {}) or {}
    jobs = spec.get("jobs", [])
    upstream = params.get("upstream") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upstream")
    ckpt_dir = params.get("weights_dir") or os.environ.get("PM_ORTHOSAM_WEIGHTS") or ""
    model_type = str(opts.get("model_type") or params.get("model_type") or "vit_b").lower()
    ckpt = os.path.join(ckpt_dir, CHECKPOINTS.get(model_type, ""))
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"SAM checkpoint missing: {ckpt}")

    import torch
    if str(params.get("devicemode") or "gpu").lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # OrthoSAM picks cuda:0 itself
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"OrthoSAM backend: mode={mode}, {len(jobs)} job(s), model={model_type}, "
        f"device={device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}, "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    log(f"options: {json.dumps(opts)}")

    # Not the temporary folder: its empty sub-folders can be cleaned away mid-run.
    work_root = os.environ.get("PM_ORTHOSAM_WORK") or os.path.join(
        os.path.expanduser("~"), ".pebblemapper", "orthosam_work")
    for ji, job in enumerate(jobs):
        path = job["path"]
        out_npz = job.get("instances_path") or (job["out_csv"] + ".instances.npz")
        t0 = time.time()
        log(f"job {ji + 1}/{len(jobs)}: {os.path.basename(path)}")
        image = _load_rgb(path, mode)
        h, w = image.shape[:2]
        log(f"  image {w}x{h} px")
        labels = _segment(image, opts, ckpt_dir, model_type, upstream, work_root)
        n = int(labels.max())
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        np.savez_compressed(out_npz, labels=labels,
                            scores=np.ones(n, dtype=np.float32),
                            shape=np.array([h, w], dtype=np.int64))
        log(f"  {n} grains -> {os.path.basename(out_npz)} ({time.time() - t0:.1f}s)")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
