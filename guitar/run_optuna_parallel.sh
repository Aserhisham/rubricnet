#!/bin/bash
for i in {1..5}
do
  /home/aser/programming/thesis/rubricnet/.venv/bin/python guitar/optuna_guitar_tuning.py --features guitar_all_v2 --n-trials 12 &
done
wait
