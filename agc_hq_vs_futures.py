"""Small AGC subset: TtbarAnalysis via FuturesExecutor vs CoffeaHQExecutor.

Uses N_FILES_MAX_PER_SAMPLE=1, USE_INFERENCE=False, and maxchunks to keep the
job small. Compares histogram bin contents.

Requires (HQ half):
  redis + TLS HQ server (started from the hq repo)
  export HQ_RESULT_DIR=/tmp/hq-results
  hq installed in the environment (pip install -e ~/irishep/hq)

Run from anywhere:
  /path/to/coffea_env/bin/python -u agc_hq_vs_futures.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.processor import FuturesExecutor

# Run relative to this repo so utils/ and the config files resolve.
HERE = Path(__file__).resolve().parent
os.chdir(HERE)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import utils  # noqa: E402
from hq.coffea import CoffeaHQExecutor  # noqa: E402
import ttbar_processor  # noqa: E402
from ttbar_processor import TtbarAnalysis  # noqa: E402

N_FILES_MAX_PER_SAMPLE = 1
USE_INFERENCE = False
USE_TRITON = False
# Keep the subset tiny while still exercising the real processor.
MAXCHUNKS = 1
CHUNKSIZE = 50_000
# One process is enough to validate hist equality; full fileset is N_FILES=1 × all samples.
SAMPLE_KEYS = ("ttbar__nominal",)

HOST = "https://localhost"
PORT = 3000
VERIFY = str(Path("~/irishep/hq/cert.pem").expanduser())


def build_fileset() -> dict:
    fileset = utils.file_input.construct_fileset(
        N_FILES_MAX_PER_SAMPLE,
        use_xcache=False,
        af_name=utils.config["benchmarking"]["AF_NAME"],
        input_from_eos=utils.config["benchmarking"]["INPUT_FROM_EOS"],
        xcache_atlas_prefix=utils.config["benchmarking"]["XCACHE_ATLAS_PREFIX"],
    )
    return {k: fileset[k] for k in SAMPLE_KEYS if k in fileset}


def run_with(executor, fileset: dict):
    NanoAODSchema.warn_missing_crossrefs = False
    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        savemetrics=True,
        metadata_cache={},
        chunksize=CHUNKSIZE,
        # maxchunks=MAXCHUNKS,
    )
    out, metrics = runner(
        fileset,
        processor_instance=TtbarAnalysis(USE_INFERENCE, USE_TRITON),
        treename="Events",
    )
    return out, metrics


def hist_snapshot(hist_dict: dict) -> dict[str, np.ndarray]:
    """Compare deterministic nominal variation only.

    Full AGC fills also include pt_res_up, which uses np.random.normal in
    utils.systematics.jet_pt_resolution — that is intentionally non-reproducible
    across processes/runs.
    """
    snap = {}
    for region, h in hist_dict.items():
        nom = h[:, :, "nominal"].project("observable")
        snap[f"{region}:values"] = np.asarray(nom.values(flow=True))
        snap[f"{region}:variances"] = np.asarray(nom.variances(flow=True))
    return snap


def assert_snaps_close(a: dict[str, np.ndarray], b: dict[str, np.ndarray], rtol=1e-6, atol=1e-6):
    assert set(a) == set(b), f"key mismatch {set(a)^set(b)}"
    for k in a:
        if not np.allclose(a[k], b[k], rtol=rtol, atol=atol, equal_nan=True):
            raise AssertionError(
                f"mismatch at {k}: max abs diff={np.nanmax(np.abs(a[k]-b[k]))}"
            )


if __name__ == "__main__":
    fileset = build_fileset()
    print("samples:", list(fileset))
    for name, info in fileset.items():
        print(f"  {name}: {len(info['files'])} file(s) -> {info['files'][0]}")

    t0 = time.monotonic()
    futures_out, _ = run_with(
        FuturesExecutor(workers=8, compression=None), fileset
    )
    futures_s = time.monotonic() - t0
    futures_snap = hist_snapshot(futures_out["hist_dict"])
    print(f"FuturesExecutor: {futures_s:.2f}s  regions={list(futures_out['hist_dict'])}")

    hq = CoffeaHQExecutor(
        host=HOST,
        port=PORT,
        verify=VERIFY,
        n_workers=8,
        queue=f"agc-hq-vs-futures-{os.getpid()}",
        poll_interval=1.0,
        pickle_modules=(utils, ttbar_processor),
        status=False,
    )
    t0 = time.monotonic()
    hq_out, _ = run_with(hq, fileset)
    hq_s = time.monotonic() - t0
    hq_snap = hist_snapshot(hq_out["hist_dict"])
    print(f"CoffeaHQExecutor: {hq_s:.2f}s  regions={list(hq_out['hist_dict'])}")

    assert_snaps_close(futures_snap, hq_snap)
    totals = {k: float(np.nansum(v)) for k, v in hq_snap.items() if k.endswith(":values")}
    print("ok: histograms match")
    print("observable value sums:", totals)
    print(
        f"timing: FuturesExecutor={futures_s:.2f}s  "
        f"CoffeaHQExecutor={hq_s:.2f}s  "
        f"ratio(HQ/Futures)={hq_s / futures_s:.2f}x"
    )
