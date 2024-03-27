#!/bin/bash

METHOD="simanneal"
TASK=$1
DEVICE=$2

if [ "$TASK" = "$WARFARIN_TASK" ]
then
  SOLVER_STEPS=2048
  SOLVER_SAMPLES=200
else
  SOLVER_STEPS=128
  SOLVER_SAMPLES=16
fi

main () {
  for SEED in 42 43 44 45 46 47 48 49; do
    python mbo/run_$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      --solver-steps $SOLVER_STEPS \
      --solver-samples $SOLVER_SAMPLES \
      --device $DEVICE
  done
}

main
