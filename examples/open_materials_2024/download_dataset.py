##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
import argparse
import logging
import os

from hydragnn.utils.datasets.download import download_file, safe_extract_tar

DOWNLOAD_LINKS = {
    "train": {
        "rattled-1000": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000.tar.gz",
        "rattled-1000-subsampled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000-subsampled.tar.gz",
        "rattled-500": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz",
        "rattled-500-subsampled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500-subsampled.tar.gz",
        "rattled-300": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz",
        "rattled-300-subsampled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz",
        "aimd-from-PBE-1000-npt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-npt.tar.gz",
        "aimd-from-PBE-1000-nvt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-nvt.tar.gz",
        "aimd-from-PBE-3000-npt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-npt.tar.gz",
        "aimd-from-PBE-3000-nvt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-nvt.tar.gz",
        "rattled-relax": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-relax.tar.gz",
    },
    "val": {
        "rattled-1000": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-1000.tar.gz",
        "rattled-1000-subsampled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-1000-subsampled.tar.gz",
        "rattled-500": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-500.tar.gz",
        "rattled-500-subsampled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-500-subsampled.tar.gz",
        "rattled-300": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-300.tar.gz",
        "rattled-300-subsampled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-300-subsampled.tar.gz",
        "aimd-from-PBE-1000-npt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-1000-npt.tar.gz",
        "aimd-from-PBE-1000-nvt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-1000-nvt.tar.gz",
        "aimd-from-PBE-3000-npt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-3000-npt.tar.gz",
        "aimd-from-PBE-3000-nvt": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-3000-nvt.tar.gz",
        "rattled-relax": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-relax.tar.gz",
    },
}


assert (
    DOWNLOAD_LINKS["train"].keys() == DOWNLOAD_LINKS["val"].keys()
), "data partition names in train do not match with equivalent names in val"
dataset_names = list(DOWNLOAD_LINKS["train"].keys())


def get_data(datadir, task, split, keep_archive=False):
    os.makedirs(datadir, exist_ok=True)

    if (task == "train" or task == "val") and split is None:
        raise NotImplementedError(f"{task} requires a split to be defined.")

    assert (
        split in DOWNLOAD_LINKS[task]
    ), f'{task}/{split}" split not defined, please specify one of the following: {list(DOWNLOAD_LINKS[task].keys())}'
    download_url = DOWNLOAD_LINKS[task][split]
    split_dir = os.path.join(datadir, task)
    os.makedirs(split_dir, exist_ok=True)

    extracted_dir = os.path.join(split_dir, split)
    is_extracted = os.path.isdir(extracted_dir) and any(
        entry.name.endswith(".aselmdb") for entry in os.scandir(extracted_dir)
    )
    if is_extracted:
        logging.info("Already extracted: %s", extracted_dir)
        return

    archive = os.path.join(split_dir, os.path.basename(download_url))
    download_file(download_url, archive)

    logging.info("Extracting contents...")
    safe_extract_tar(archive, split_dir)
    if not keep_archive:
        os.remove(archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./dataset",
        help="Specify path to save datasets. Defaults to './dataset'",
    )
    parser.add_argument("--task", choices=["train", "val"])
    parser.add_argument("--split", choices=dataset_names)
    parser.add_argument("--keep-archives", action="store_true")

    args, _ = parser.parse_known_args()

    tasks = [args.task] if args.task else ["train", "val"]
    splits = [args.split] if args.split else dataset_names
    for task in tasks:
        for split in splits:
            get_data(
                datadir=args.data_path,
                task=task,
                split=split,
                keep_archive=args.keep_archives,
            )
