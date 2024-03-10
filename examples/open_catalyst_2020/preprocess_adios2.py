"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import argparse
import glob
import os

import ase.io
import numpy as np
import torch
from tqdm import tqdm
from mpi4py import MPI

from atoms_to_graphs import AtomsToGraphs

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import hydragnn


def write_images_to_adios(a2g, samples, pid, args, comm):
    fname = os.path.join("./data", args.out_path + ".bp")
    adwriter = AdiosWriter(fname, comm)

    pbar = tqdm(
        total=5000 * len(samples),
        position=pid,
        desc="Preprocessing data into ADIOS",
    )

    trainset = []
    idx = 0

    for sample in samples:
        traj_logs = open(sample, "r").read().splitlines()
        xyz_idx = os.path.splitext(os.path.basename(sample))[0]
        traj_path = os.path.join(args.data_path, f"{xyz_idx}.extxyz")
        traj_frames = ase.io.read(traj_path, ":")

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
            if args.ref_energy and not args.test_data:
                ref_energy = float(frame_log[2])
                data_object.y -= torch.FloatTensor([ref_energy])

            trainset.append(data_object)
            idx += 1
            pbar.update(1)

    adwriter.add("dataset", trainset)
    adwriter.save()


def main(args):
    xyz_logs = glob.glob(os.path.join(args.data_path, "*.txt"))
    if not xyz_logs:
        raise RuntimeError("No *.txt files found. Did you uncompress?")

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
    )

    comm = MPI.COMM_WORLD
    comm_size, comm_rank = hydragnn.utils.setup_ddp()

    # Chunk the trajectories into args.num_workers splits
    chunked_txt_files = np.array_split(xyz_logs, comm_size)

    write_images_to_adios(a2g, chunked_txt_files[comm_rank], comm_rank, args, comm)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to dir containing *.extxyz and *.txt files",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)