import argparse
import gzip
import json
import logging
import os
import shutil
import urllib.parse
import urllib.request


HF_DATASET_API = "https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1"
HF_DATASET_FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _hf_tree(repo: str):
    url = HF_DATASET_API.format(repo=urllib.parse.quote(repo, safe="/"))
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _hf_case_paths(repo: str):
    tree = _hf_tree(repo)
    cases = set()
    for entry in tree:
        path = entry.get("path", "")
        if path.endswith("/case.json.gz"):
            cases.add(path.split("/")[0])
    return sorted(cases)


def discover_cases(root: str, repo: str):
    del root
    return _hf_case_paths(repo)


def _download_file(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.part"
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp_path, "wb") as fh:
        shutil.copyfileobj(response, fh)
    os.replace(tmp_path, out_path)


def _gunzip_file(src_gz: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = f"{dst_path}.part"
    with gzip.open(src_gz, "rb") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp_path, dst_path)


def ensure_pglearn_downloaded(
    root: str,
    repo: str,
    case_name: str,
    formulation: str,
    rank: int,
    comm,
    splits=("train", "test"),
    uncompress_h5=True,
):
    if rank == 0:
        os.makedirs(root, exist_ok=True)

        required_files = [
            f"{case_name}/case.json.gz",
            f"{case_name}/config.toml",
        ]
        for split in splits:
            required_files.extend(
                [
                    f"{case_name}/{split}/input.h5.gz",
                    f"{case_name}/{split}/{formulation}/primal.h5.gz",
                ]
            )

        for rel_path in required_files:
            local_path = os.path.join(root, rel_path)
            if not os.path.isfile(local_path):
                url = HF_DATASET_FILE.format(
                    repo=urllib.parse.quote(repo, safe="/"),
                    path=urllib.parse.quote(rel_path, safe="/"),
                )
                logging.info("Downloading %s", rel_path)
                _download_file(url, local_path)

            if uncompress_h5 and local_path.endswith(".h5.gz"):
                uncompressed = local_path[:-3]
                if not os.path.isfile(uncompressed):
                    logging.info("Uncompressing %s", rel_path)
                    _gunzip_file(local_path, uncompressed)

    comm.Barrier()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo", type=str, default="PGLearn/PGLearn-Small")
    parser.add_argument("--case_name", type=str, default="14_ieee")
    parser.add_argument("--formulation", type=str, default="ACOPF")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--no_uncompress_h5", action="store_true")
    return parser.parse_args()


def main():
    from mpi4py import MPI

    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_root)
    ensure_pglearn_downloaded(
        root=root,
        repo=args.repo,
        case_name=args.case_name,
        formulation=args.formulation,
        rank=rank,
        comm=comm,
        uncompress_h5=(not args.no_uncompress_h5),
    )

    if rank == 0:
        logging.info("PGLearn data is ready under %s", root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
