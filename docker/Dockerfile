ARG BASE=nvidia/cuda:10.2-devel
FROM $BASE

ARG NPROCS=4

RUN apt-get update && apt-get install -y \
        bc \
        git \
        wget \
        python3 \
        python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN python3 -m pip install pickle5 matplotlib torch>=1.8 tqdm tensorboard && \
    python3 -m pip install torch-geometric>=1.7.2 torch-scatter torch-sparse && \
    python3 -m pip install pytest
