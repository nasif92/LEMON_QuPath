#!/usr/bin/env bash
# LEMON + QuPath vs QuPath alone, 3 seeds each. Usage:  bash run_experiments.sh Ki67   (default Ki67)
# Finished runs (log has a RESULT line) are skipped, so re-running resumes.
# The labelled-binary test set (millions of nuclei) is skipped unless you run:  LABELLED=1 bash run_experiments.sh Ki67
set -u
ST=${1:-Ki67}
SEEDS=${SEEDS:-"42 43 44"}
mkdir -p runs
if [ -z "${LABELLED:-}" ]; then EXTRA="--no_labelled"; else EXTRA=""; fi

run() {  # run <name> <train.py args...>
  local name=$1; shift
  for s in $SEEDS; do
    local log="runs/${ST}_${name}_s${s}.log"
    if [ -f "$log" ] && grep -q "^RESULT " "$log"; then echo "skip $log"; continue; fi
    echo "run  $log"
    python train.py --subtypes "$ST" --seed "$s" $EXTRA "$@" > "$log" 2>&1 \
      || echo "FAILED $log"
  done
}

run qupath_all          --input qupath --feature_set all_derived            # baseline (108)
run lemon               --input lemon                                       # LEMON alone
run fusion_all          --input fusion --feature_set all_derived            # 108 + LEMON
run fusion_all_pca32    --input fusion --feature_set all_derived --lemon_pca 32
run qupath_context      --input qupath --feature_set context_only_derived   # neighbourhood only (44)
run fusion_context      --input fusion --feature_set context_only_derived   # can LEMON replace nucleus features?

run qupath_nocontext    --input qupath --feature_set no_context_derived     # own-nucleus features only (64)

python summarize.py runs/${ST}_*.log