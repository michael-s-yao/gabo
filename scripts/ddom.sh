#!/bin/bash

METHOD="ddom"
TASK=$1
DEVICE=$2
BUDGET=2048

main () {
  for SEED in 42 43 44 45 46; do
    python mbo/run_$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      --budget $BUDGET \
      --device $DEVICE
  done
}

main
