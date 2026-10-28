#!/bin/bash
# Bind one process to one GPU by local rank.
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
exec "$@"
