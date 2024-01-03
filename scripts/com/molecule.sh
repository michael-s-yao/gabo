#!/bin/bash

METHOD="com"
TASK=$MOLECULE_TASK
PARTICLE_GRADIENT_STEPS=32
EVALUATION_SAMPLES=8

main () {
  for SEED in 42 43 44 45 46; do
    python mbo/$METHOD.py \
      --seed $SEED \
      --task $TASK \
      --particle-train-gradient-steps $PARTICLE_GRADIENT_STEPS \
      --particle-evaluate-gradient-steps $PARTICLE_GRADIENT_STEPS \
      --evaluation-samples $EVALUATION_SAMPLES \
      --logging-dir db-results/$METHOD-$TASK-$SEED \
      --not-fast
  done
}

main
