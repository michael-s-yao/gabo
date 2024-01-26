#!/bin/bash

METHOD="bdi"
SEEDS="42 43 44 45 46"
TASK=$1
BUDGET=2048

python mbo/run_$METHOD.py --task $TASK --seeds $SEEDS --budget $BUDGET
