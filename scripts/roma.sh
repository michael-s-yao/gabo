#!/bin/bash

METHOD="roma"
TASK=$1
BUDGET=2048

main () {
  for SEED in 42 43 44 45 46 47 48 49; do
    python mbo/run_$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      --budget $BUDGET
  done
}

main
