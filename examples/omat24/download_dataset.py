import argparse
import glob
import logging
import os
import shutil


DOWNLOAD_LINKS = {
    "train": {
        "rattled-1000": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000.tar.gz",
        "rattled-1000-subsampled": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000-subsampled.tar.gz",
        "rattled-500": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz",
        "rattled-500-subsampled": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500-subsampled.tar.gz",
        "rattled-300": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz",
        "rattled-300-subsampled": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz",
        "aimd-from-PBE-1000-npt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-npt.tar.gz",
        "aimd-from-PBE-1000-nvt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-nvt.tar.gz",
        "aimd-from-PBE-3000-npt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-npt.tar.gz",
        "aimd-from-PBE-3000-nvt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-nvt.tar.gz",
        "rattled-relax": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-relax.tar.gz",
    },
    "val": {
        "rattled-1000": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-1000.tar.gz",
        "rattled-1000-subsampled": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-1000-subsampled.tar.gz",
        "rattled-500": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-500.tar.gz",
        "rattled-500-subsampled": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-500-subsampled.tar.gz",
        "rattled-300": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-300.tar.gz",
        "rattled-300-subsampled": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-300-subsampled.tar.gz",
        "aimd-from-PBE-1000-npt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-1000-npt.tar.gz",
        "aimd-from-PBE-1000-nvt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-1000-nvt.tar.gz",
        "aimd-from-PBE-3000-npt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-3000-npt.tar.gz",
        "aimd-from-PBE-3000-nvt": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/aimd-from-PBE-3000-nvt.tar.gz",
        "rattled-relax": "wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-relax.tar.gz",
    },
}


assert (
    DOWNLOAD_LINKS["train"].keys() == DOWNLOAD_LINKS["val"].keys()
), "data partition names in train do not match with equivalent names in val"
dataset_names = list(DOWNLOAD_LINKS["train"].keys())


def get_data(datadir, task, split):
    os.makedirs(datadir, exist_ok=True)

    if (task == "train" or task == "val") and split is None:
        raise NotImplementedError(f"{task} requires a split to be defined.")

    assert (
        split in DOWNLOAD_LINKS[task]
    ), f'{task}/{split}" split not defined, please specify one of the following: {list(DOWNLOAD_LINKS[task].keys())}'
    download_link = DOWNLOAD_LINKS[task][split]

    os.system(f"wget {download_link} -P {datadir}")
    filename = os.path.join(datadir, os.path.basename(download_link))
    logging.info("Extracting contents...")
    os.system(f"tar -xvf {filename} -C {datadir}")
    filename_without_extension = filename.replace(".tar.gz", "")
    os.makedirs(os.path.join(datadir, task), exist_ok=False)

    # Move the directory
    shutil.move(filename_without_extension, os.path.join(datadir, task))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./dataset",
        help="Specify path to save datasets. Defaults to './dataset'",
    )

    args, _ = parser.parse_known_args()

    for task in ["train", "val"]:
        for split in dataset_names:
            get_data(
                datadir=args.data_path,
                task=task,
                split=split,
            )
