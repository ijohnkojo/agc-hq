"""Small AGC subset: local hists via FuturesExecutor vs histserv via CoffeaHQExecutor.

Same shape as agc_hq_vs_futures.py, but the HQ run streams fills to a histserv
server (workers fill remotely, client snapshots at the end) instead of
returning pickled histograms over the shared filesystem.

Requires (HQ half):
  redis + HQ server (started from the hq repo; plain HTTP, no TLS)
  histserv --port 50051
  export HQ_RESULT_DIR=/tmp/hq-results
  hq + histserv installed in the environment (pip install -e ~/irishep/hq histserv)

Run from anywhere:
  /path/to/coffea_env/bin/python -u agc_histserv_vs_futures.py
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
from hq.histserv import init_remote_hists, snapshot_hists  # noqa: E402
import ttbar_processor  # noqa: E402
from ttbar_processor import TtbarAnalysis, make_hist_templates  # noqa: E402

N_FILES_MAX_PER_SAMPLE = 1
USE_INFERENCE = False
USE_TRITON = False
CHUNKSIZE = 50_000
# Keep the subset tiny while still exercising the real processor + histserv path.
MAXCHUNKS = 1
SAMPLE_KEYS = ("ttbar__nominal",)

HOST = "http://localhost"
PORT = 3000
HISTSERV_ADDRESS = "localhost:50051"


def build_fileset() -> dict:
    fileset = utils.file_input.construct_fileset(
        N_FILES_MAX_PER_SAMPLE,
        use_xcache=False,
        af_name=utils.config["benchmarking"]["AF_NAME"],
        input_from_eos=utils.config["benchmarking"]["INPUT_FROM_EOS"],
        xcache_atlas_prefix=utils.config["benchmarking"]["XCACHE_ATLAS_PREFIX"],
    )
    return {k: fileset[k] for k in SAMPLE_KEYS if k in fileset}


def run_with(executor, fileset: dict, remote_hists=None):
    NanoAODSchema.warn_missing_crossrefs = False
    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        savemetrics=True,
        metadata_cache={},
        chunksize=CHUNKSIZE,
        maxchunks=MAXCHUNKS,
    )
    out, metrics = runner(
        fileset,
        processor_instance=TtbarAnalysis(
            USE_INFERENCE, USE_TRITON, remote_hists=remote_hists
        ),
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
        # index by axis name: the histserv round-trip may reorder axes
        nom = h[{"variation": "nominal"}].project("observable")
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

    # reference: local hists, merged client-side by coffea accumulate
    t0 = time.monotonic()
    futures_out, _ = run_with(
        FuturesExecutor(workers=8, compression=None), fileset
    )
    futures_s = time.monotonic() - t0
    futures_snap = hist_snapshot(futures_out["hist_dict"])
    print(f"FuturesExecutor (local hists): {futures_s:.2f}s  regions={list(futures_out['hist_dict'])}")

    # histserv: init once, workers fill remotely, snapshot merged bins at the end
    remote_hists = init_remote_hists(make_hist_templates(), address=HISTSERV_ADDRESS)
    hq = CoffeaHQExecutor(
        host=HOST,
        port=PORT,
        n_workers=8,
        queue=f"agc-histserv-vs-futures-{os.getpid()}",
        poll_interval=1.0,
        pickle_modules=(utils, ttbar_processor),
        status=False,
    )
    t0 = time.monotonic()
    hq_out, _ = run_with(hq, fileset, remote_hists=remote_hists)
    hq_s = time.monotonic() - t0
    hist_dict = snapshot_hists(remote_hists, delete_from_server=True)
    hq_snap = hist_snapshot(hist_dict)
    print(f"CoffeaHQExecutor+histserv: {hq_s:.2f}s  regions={list(hist_dict)}")
    print(f"  task results were tiny: {hq_out}")

    assert_snaps_close(futures_snap, hq_snap)
    totals = {k: float(np.nansum(v)) for k, v in hq_snap.items() if k.endswith(":values")}
    print("ok: histograms match")
    print("observable value sums:", totals)
    print(
        f"timing: FuturesExecutor={futures_s:.2f}s  "
        f"CoffeaHQExecutor+histserv={hq_s:.2f}s  "
        f"ratio={hq_s / futures_s:.2f}x"
    )
