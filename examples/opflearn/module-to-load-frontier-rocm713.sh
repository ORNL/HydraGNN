#!/bin/bash
# Frontier module stack validated with ROCm 7.13 workflows.
module reset
ml PrgEnv-gnu/8.7.0
ml cpe/26.03
ml miniforge3/23.11.0-0
ml rocm/7.13.0
ml rccl-net-plugin
ml craype-accel-amd-gfx90a
ml git-lfs
module unload darshan-runtime || true
