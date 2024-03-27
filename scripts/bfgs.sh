#!/bin/bash

METHOD="bfgs"
TASK=$1
DEVICE=$2

if [ "$TASK" = "$WARFARIN_TASK" ]
then
  PARTICLE_GRADIENT_STEPS=2048
  EVALUATION_SAMPLES=200
else
  PARTICLE_GRADIENT_STEPS=128
  EVALUATION_SAMPLES=16
fi

main () {
  for SEED in 42 43 44 45 46 47 48 49; do
    python mbo/run_$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      --solver-steps $PARTICLE_GRADIENT_STEPS \
      --solver-samples $EVALUATION_SAMPLES \
      --device $DEVICE
  done
}

main
