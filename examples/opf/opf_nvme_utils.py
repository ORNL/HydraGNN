import logging
import os
import shutil
import subprocess
import tarfile
import time
from mpi4py import MPI


def opf_release_name(topological_perturbations: bool) -> str:
    return (
        "dataset_release_1_nminusone"
        if topological_perturbations
        else "dataset_release_1"
    )


def find_nvme_root(preferred_root: str | None) -> str | None:
    candidates = []
    if preferred_root:
        candidates.append(preferred_root)
    user = os.getenv("USER", "")
    if user:
        candidates.append(f"/mnt/bb/{user}")
    for env_var in ["LOCAL_SCRATCH", "SLURM_TMPDIR", "TMPDIR"]:
        value = os.getenv(env_var)
        if value:
            candidates.append(value)

    for root in candidates:
        if os.path.isdir(root) and os.access(root, os.W_OK | os.X_OK):
            return root
    return None


def stage_case_to_nvme(
    source_datadir: str,
    case_name: str,
    topological_perturbations: bool,
    comm,
    rank: int,
    nvme_root: str | None,
    serialized_targets: list[str] | None = None,
) -> str:
    selected_nvme_root = find_nvme_root(nvme_root)
    if selected_nvme_root is None:
        if rank == 0:
            logging.warning(
                "NVMe requested but no writable local scratch path found; using shared datadir=%s",
                source_datadir,
            )
        return source_datadir

    release = opf_release_name(topological_perturbations)
    src_case_dir = os.path.join(source_datadir, release, case_name)
    src_raw_dir = os.path.join(src_case_dir, "raw")
    if not os.path.isdir(src_case_dir):
        if rank == 0:
            logging.warning(
                "NVMe staging skipped: source case directory not found (%s)",
                src_case_dir,
            )
        return source_datadir

    if not os.path.isdir(src_raw_dir):
        if rank == 0:
            logging.warning(
                "NVMe staging skipped: source raw directory not found (%s)",
                src_raw_dir,
            )
        return source_datadir

    archive_names = sorted(
        name
        for name in os.listdir(src_raw_dir)
        if name.startswith(f"{case_name}_") and name.endswith(".tar.gz")
    )
    if not archive_names:
        if rank == 0:
            logging.warning(
                "NVMe staging skipped: no OPF archives found in %s",
                src_raw_dir,
            )
        return source_datadir

    job_id = os.getenv("SLURM_JOB_ID", "interactive")
    staged_datadir = os.path.join(selected_nvme_root, "hydragnn_opf", job_id, "dataset")
    dst_case_dir = os.path.join(staged_datadir, release, case_name)
    dst_raw_dir = os.path.join(dst_case_dir, "raw")
    copy_marker = os.path.join(dst_case_dir, ".hydragnn_archives_copied")
    extract_marker = os.path.join(dst_case_dir, ".hydragnn_archives_extracted")

    effective_refresh = False
    if serialized_targets:
        missing_targets = [
            rel_path
            for rel_path in serialized_targets
            if not os.path.exists(os.path.join(staged_datadir, rel_path))
        ]
        if missing_targets:
            effective_refresh = True
            if rank == 0:
                logging.info(
                    "Forcing clean NVMe restore for case=%s because serialized output is missing: %s",
                    case_name,
                    ", ".join(missing_targets),
                )

    tar_path = shutil.which("tar")

    def _extract_archive(archive_path: str, dst_dir: str):
        if tar_path is not None:
            subprocess.run([tar_path, "-xzf", archive_path, "-C", dst_dir], check=True)
            return
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dst_dir)

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, rank)
    local_rank = local_comm.Get_rank()
    local_ok = True
    local_err = ""

    if local_rank == 0:
        try:
            os.makedirs(dst_raw_dir, exist_ok=True)
            if effective_refresh and os.path.isdir(dst_case_dir):
                shutil.rmtree(dst_case_dir, ignore_errors=True)
                os.makedirs(dst_raw_dir, exist_ok=True)

            if not os.path.isfile(copy_marker):
                t0_copy = time.perf_counter()
                logging.info(
                    "Copying OPF archives to NVMe: src=%s dst=%s archives=%d",
                    src_raw_dir,
                    dst_raw_dir,
                    len(archive_names),
                )
                for archive_name in archive_names:
                    src_archive = os.path.join(src_raw_dir, archive_name)
                    dst_archive = os.path.join(dst_raw_dir, archive_name)
                    if os.path.isfile(dst_archive) and os.path.getsize(
                        dst_archive
                    ) == os.path.getsize(src_archive):
                        continue
                    shutil.copy2(src_archive, dst_archive)
                with open(copy_marker, "w") as marker:
                    marker.write(f"archives={len(archive_names)}\n")
                logging.info(
                    "OPF archive copy complete for case=%s elapsed=%.1fs",
                    case_name,
                    time.perf_counter() - t0_copy,
                )

            if not os.path.isfile(extract_marker):
                t0_extract = time.perf_counter()
                logging.info(
                    "Extracting OPF archives on NVMe: case=%s archives=%d",
                    case_name,
                    len(archive_names),
                )
                for archive_name in archive_names:
                    dst_archive = os.path.join(dst_raw_dir, archive_name)
                    _extract_archive(dst_archive, dst_raw_dir)
                with open(extract_marker, "w") as marker:
                    marker.write(f"archives={len(archive_names)}\n")
                logging.info(
                    "OPF archive extraction complete for case=%s elapsed=%.1fs",
                    case_name,
                    time.perf_counter() - t0_extract,
                )
        except Exception as exc:
            local_ok = False
            local_err = str(exc)

    local_ok = local_comm.bcast(local_ok, root=0)
    local_err = local_comm.bcast(local_err, root=0)
    all_ok = comm.allreduce(1 if local_ok else 0, op=MPI.MIN) == 1
    local_comm.Barrier()
    local_comm.Free()
    comm.Barrier()

    if not all_ok:
        if rank == 0:
            logging.warning(
                "NVMe staging failed on at least one node (%s); falling back to shared datadir=%s",
                local_err,
                source_datadir,
            )
        return source_datadir

    if rank == 0:
        logging.info("Using node-local staged datadir: %s", staged_datadir)
    return staged_datadir
