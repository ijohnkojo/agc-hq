"""Outside-notebook coffea Runner smoke: 1 NanoAOD file via CoffeaHQExecutor.

Compares event counts to FuturesExecutor. No full AGC TtbarAnalysis.

Requires redis + TLS HQ server (started from the hq repo), and:
  export HQ_RESULT_DIR=/tmp/hq-results
  hq installed in the environment (pip install -e ~/irishep/hq)

Run from anywhere:
  /path/to/coffea_env/bin/python -u coffea_hq_runner_smoke.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.processor import FuturesExecutor

from hq.coffea import CoffeaHQExecutor

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "nanoaod_inputs.json"

HOST = "https://localhost"
PORT = 3000
VERIFY = str(Path("~/irishep/hq/cert.pem").expanduser())

# Keep the job tiny: one file, one chunk.
CHUNKSIZE = 10_000
MAXCHUNKS = 1


class CountEvents(processor.ProcessorABC):
    """Minimal processor: count events per dataset."""

    def process(self, events):
        dataset = events.metadata["dataset"]
        return {"nevents": {dataset: len(events)}}

    def postprocess(self, accumulator):
        return accumulator


def one_file_fileset() -> dict:
    info = json.loads(INPUTS.read_text())
    path = info["ttbar"]["nominal"]["files"][0]["path"]
    return {
        "ttbar__nominal": {
            "files": [path],
            "metadata": {
                "process": "ttbar",
                "variation": "nominal",
            },
        }
    }


def run_with(executor: processor.executor.ExecutorBase, fileset: dict):
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
        fileset, processor_instance=CountEvents(), treename="Events"
    )
    return out, metrics


if __name__ == "__main__":
    fileset = one_file_fileset()
    print("file:", fileset["ttbar__nominal"]["files"][0])

    futures_out, futures_metrics = run_with(
        FuturesExecutor(workers=2, compression=None), fileset
    )
    futures_nevents = futures_out["nevents"]
    print("FuturesExecutor:", futures_nevents)

    hq_executor = CoffeaHQExecutor(
        host=HOST,
        port=PORT,
        verify=VERIFY,
        n_workers=2,
        queue=f"coffea-hq-runner-smoke-{os.getpid()}",
        poll_interval=1.0,
        status=False,
    )
    hq_out, hq_metrics = run_with(hq_executor, fileset)
    hq_nevents = hq_out["nevents"]
    print("CoffeaHQExecutor:", hq_nevents)

    assert hq_nevents == futures_nevents, f"mismatch: {hq_nevents=} {futures_nevents=}"
    print("ok: nevents match", hq_nevents)
    print("hq metrics entries:", sorted(hq_metrics.keys()) if hq_metrics else None)
