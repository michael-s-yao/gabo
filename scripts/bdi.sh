#!/bin/bash

METHOD="bdi"
SEEDS="42 43 44 45 46"
TASK=$1

python mbo/run_$METHOD.py $TASK --seeds $SEEDS
