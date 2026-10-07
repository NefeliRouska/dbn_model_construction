#!/bin/sh
# Long jobs meant for a machine that can run unattended for a day or more.
#
#   sh scripts/run_long_jobs.sh sweep      full pgmpy sweep at 1 s      (~20 h per dataset, one process each)
#   sh scripts/run_long_jobs.sh studies    every ablation.py study + the stand-alone studies (~1.5 h per dataset)
#   sh scripts/run_long_jobs.sh tabpfn     TabPFN comparison, version $TABPFN_VERSION (default v3.5)
#   sh scripts/run_long_jobs.sh all
#
# Before starting:
#   - python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
#   - data/dbn_wide_*.csv present (python data_prep/convert_prom_dump.py --csv data/prom_dump_all_<stem>.csv)
#   - commit first: every output file carries the commit hash, "-dirty" otherwise
#   - for tabpfn: `hf auth login` (Hugging Face, after accepting the terms of
#     Prior-Labs/tabpfn_3_5) AND the Prior Labs licence (https://ux.priorlabs.ai,
#     or let the first run open the browser). If Python reports a certificate
#     error, the SSL_CERT_FILE line below takes care of it.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WHAT=${1:-all}
TABPFN_VERSION=${TABPFN_VERSION:-v3.5}
STAMP=$(date +%Y%m%d_%H%M%S)
export SSL_CERT_FILE=$($PY -m certifi 2>/dev/null || true)
mkdir -p results/logs

if [ "$WHAT" = sweep ] || [ "$WHAT" = all ]; then
  for csv in data/dbn_wide_*.csv; do
    stem=$(basename "$csv" .csv | sed 's/dbn_wide_//')
    nohup $PY dbn/dbn_sweep.py --csv "$csv" --granularity 1 --n-bins 10 20 30 \
      > "results/logs/dbn_${stem}_1s_${STAMP}.log" 2>&1 &
  done
  echo "sweeps started in background; logs in results/logs/"
fi

if [ "$WHAT" = studies ] || [ "$WHAT" = all ]; then
  for csv in data/dbn_wide_*.csv; do
    for study in $($PY dbn/ablation.py --list-studies 2>/dev/null | cut -d: -f1); do
      $PY dbn/ablation.py --csv "$csv" --study "$study"
    done
    $PY dbn/ablation.py --csv "$csv" --study refs --with-hgb
    $PY dbn/rollout.py --csv "$csv"
    $PY analysis/regime_study.py --csv "$csv"
    $PY analysis/adaptation_study.py --csv "$csv"
    $PY analysis/whatif_study.py --csv "$csv"
  done
  $PY analysis/transfer_study.py --csv data/dbn_wide_*.csv
  for study in $($PY dbn/ablation.py --list-studies 2>/dev/null | cut -d: -f1); do
    $PY analysis/ablation_report.py --study "$study" --ref-config self > /dev/null
  done
  $PY analysis/ablation_report.py --study ofat > /dev/null
  $PY analysis/ablation_report.py --study constraints > /dev/null
fi

if [ "$WHAT" = tabpfn ] || [ "$WHAT" = all ]; then
  for csv in data/dbn_wide_*.csv; do
    $PY analysis/tabpfn_baseline.py --csv "$csv" --model-version "$TABPFN_VERSION"
    $PY analysis/whatif_study.py --csv "$csv" --tabpfn "$TABPFN_VERSION"
  done
  $PY analysis/tabpfn_report.py --version "$TABPFN_VERSION"
fi
