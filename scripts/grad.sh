#!/bin/bash

METHOD="grad"
TASK=$1

if [ "$TASK" = "$WARFARIN_TASK" ]
then
  PARTICLE_GRADIENT_STEPS=2048
  EVALUATION_SAMPLES=1
else
  PARTICLE_GRADIENT_STEPS=128
  EVALUATION_SAMPLES=16
fi

main () {
  for AGGREGATION_METHOD in "None" "mean" "min"; do
    for SEED in 42 43 44 45 46 47 48 49; do
      python mbo/run_$METHOD.py \
        --seed $SEED \
        --task $TASK \
        --logging-dir db-results/$METHOD-$AGGREGATION_METHOD-$TASK-$SEED \
        --aggregation-method $AGGREGATION_METHOD \
        --particle-evaluate-gradient-steps $PARTICLE_GRADIENT_STEPS \
        --evaluation-samples $EVALUATION_SAMPLES
    done
  done
}

main
