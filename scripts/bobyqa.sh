#!/bin/bash

METHOD="bobyqa"
TASK=$1
DEVICE=$2

main () {
  for SEED in 42 43 44 45 46; do
    python mbo/run_$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      --device $DEVICE
  done
}

main
