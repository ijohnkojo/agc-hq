# agc-hq

CMS Open Data $t\bar{t}$ analysis (from the IRIS-HEP [Analysis Grand Challenge](https://github.com/iris-hep/analysis-grand-challenge)) used as an end-to-end test of [hq](../hq), a pull-based task queue with a coffea executor.

This repo holds only analysis code. `hq` itself lives in `../hq` and is consumed as a normal installed Python package — no `sys.path` hacks.

## Layout

| Path | What it is |
| --- | --- |
| `ttbar_analysis_pipeline.ipynb` | Full AGC ttbar pipeline; set `USE_HQ = True` to run chunks through hq |
| `ttbar_analysis_pipeline_futures.ipynb` | Same pipeline pinned to coffea's `FuturesExecutor` (reference/baseline) |
| `ttbar_processor.py` | The `TtbarAnalysis` coffea processor |
| `utils/` | AGC helper package (fileset construction, plotting, ML, systematics, config) |
| `agc_hq_vs_futures.py` | Runs a small AGC subset on both executors and asserts the histograms match |
| `coffea_hq_runner_smoke.py` | Minimal coffea `Runner` smoke test (event counting, 1 file / 1 chunk) |
| `corrections.json`, `nanoaod_inputs.json` | Analysis inputs |
| `cabinetry_config.yml`, `cabinetry_config_ml.yml` | cabinetry fit configurations |
| `histograms/`, `figures/`, `metrics/`, `histograms.root`, `workspace.json` | Generated outputs (safe to delete) |

## Setup

Everything runs in the `coffea_env` conda environment. One-time setup:

```bash
conda activate coffea_env

# analysis dependencies (most are already in coffea_env)
pip install -r requirements.txt

# hq as an editable package — changes in ../hq apply immediately
pip install -e ../hq
```

After this, `from hq.coffea import CoffeaHQExecutor` works from any directory, with no `PYTHONPATH` needed.

## Running against hq

The hq services still live in the hq repo. In one terminal:

```bash
cd ../hq
# redis + TLS server; see hq's README / docs for details
bash scripts/testrun.sh   # or start redis + `bun typescript/server.ts` manually
export HQ_RESULT_DIR=/tmp/hq-results
```

Then, from this repo:

```bash
python -u agc_hq_vs_futures.py          # HQ vs Futures histogram comparison
python -u coffea_hq_runner_smoke.py     # minimal Runner smoke test
jupyter lab ttbar_analysis_pipeline.ipynb
```

The notebooks and scripts point at the server's TLS certificate via `~/irishep/hq/cert.pem` (`HQ_VERIFY` in the notebook config cell). Adjust it if the hq repo lives elsewhere.

## Knobs that matter

- `N_FILES_MAX_PER_SAMPLE` — files per sample; `1` for a quick run
- `HQ_N_WORKERS` — managed hq workers (roughly one per core)
- `maxchunks` on `processor.Runner` — cap chunks per file for smoke tests
- `USE_INFERENCE` — ML inference; leave `False` unless the BDT models are available
