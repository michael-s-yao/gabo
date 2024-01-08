#!/bin/bash

METHOD="grad"
TASK=$1

main () {
  for AGGREGATION_METHOD in "None" "mean" "min"; do
    for SEED in 42 43 44 45 46; do
      python mbo/run_$METHOD.py \
        --seed $SEED \
        --task $TASK \
        --logging-dir db-results/$METHOD-$AGGREGATION_METHOD-$TASK-$SEED \
        --aggregation-method $AGGREGATION_METHOD
    done
  done
}

main
