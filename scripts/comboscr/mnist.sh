#!/bin/bash

METHOD="comboscr"
TASK=$MNIST_TASK
DEVICE="cuda:1"

run () {
  if [ -z "$1" ]
  then
    ALPHA=""
  else
    ALPHA="--alpha $1"
  fi
  for SEED in 42 43 44 45 46; do
    python mbo/$METHOD.py \
      --task $TASK \
      --logging-dir db-results/$METHOD-$TASK-$1-$SEED \
      --seed $SEED \
      --device $DEVICE \
      $ALPHA
  done
}

main () {
  run
  for CONSTANT in 0.0 0.2 0.5 0.8 1.0; do
    run $CONSTANT
  done
}

main