#!/bin/bash

METHOD="bonet"
TASK=$1
MODE="both"

main () {
  for SEED in 42 43 44 45 46; do
    python mbo/run_$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      --ckpt-dir checkpoints/$METHOD/$TASK \
      --mode $MODE
  done
}

main
