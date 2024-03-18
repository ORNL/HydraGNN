"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os

import ase.io
import torch
from tqdm import tqdm


def write_images_to_adios(a2g, samples, data_path, subtract_reference_energy=False):

    dataset = []
    idx = 0

    rank = torch.distributed.get_rank()
    for sample in tqdm(
        samples, desc=os.path.basename(data_path), disable=False if rank == 0 else True
    ):
        try:
            traj_logs = open(sample, "r").read().splitlines()
            xyz_idx = os.path.splitext(os.path.basename(sample))[0]
            traj_path = os.path.join(data_path, f"{xyz_idx}.extxyz")
            traj_frames = ase.io.read(traj_path, ":", parallel=False)
            ## Check length consistency (got error with 199.txt)
            if len(traj_logs) != len(traj_frames):
                print(rank, "skip:", sample)
                continue

        except Exception as e:
            print(f"WARN:", type(error).__name__)
            continue

        for i, frame in enumerate(traj_frames):
            frame_log = traj_logs[i].split(",")
            sid = int(frame_log[0].split("random")[1])
            fid = int(frame_log[1].split("frame")[1])
            data_object = a2g.convert(frame)
            # add atom tags
            data_object.tags = torch.LongTensor(frame.get_tags())
            data_object.sid = torch.IntTensor([sid])
            data_object.fid = torch.IntTensor([fid])

            # subtract off reference energy
            if subtract_reference_energy:
                ref_energy = float(frame_log[2])
                data_object.y -= torch.FloatTensor([ref_energy])

            dataset.append(data_object)
            idx += 1

    return dataset
