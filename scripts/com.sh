#!/bin/bash

METHOD="com"
TASK=$1

if [ "$TASK" = "$WARFARIN_TASK" ]
then
  PARTICLE_GRADIENT_STEPS=2048
  EVALUATION_SAMPLES=1
else
  PARTICLE_GRADIENT_STEPS=128
  EVALUATION_SAMPLES=16

if [ "$TASK" = "$CHEMBL_TASK" ]
then
  DO_TASK_RELABEL="--no-task-relabel"
else
  DO_TASK_RELABEL="--task-relabel"
fi

main () {
  for SEED in 42 43 44 45 46 47 48 49; do
    python mbo/run_$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --particle-train-gradient-steps $PARTICLE_GRADIENT_STEPS \
      --particle-evaluate-gradient-steps $PARTICLE_GRADIENT_STEPS \
      --evaluation-samples $EVALUATION_SAMPLES \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      $DO_TASK_RELABEL \
      --not-fast
  done
}

main
