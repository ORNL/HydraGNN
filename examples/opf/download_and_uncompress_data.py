import logging
import os
import shutil
import socket
import urllib.error
import urllib.request
from contextlib import contextmanager

import torch_geometric.datasets.opf as tg_opf


def _opf_release_name(topological_perturbations: bool) -> str:
    return (
        "dataset_release_1_nminusone"
        if topological_perturbations
        else "dataset_release_1"
    )


def _opf_raw_dir(root: str, case_name: str, topological_perturbations: bool) -> str:
    return os.path.join(
        root, _opf_release_name(topological_perturbations), case_name, "raw"
    )


def _opf_release_dir(root: str, topological_perturbations: bool) -> str:
    return os.path.join(root, _opf_release_name(topological_perturbations))


def _opf_tmp_dir(root: str, case_name: str, topological_perturbations: bool) -> str:
    return os.path.join(
        _opf_raw_dir(root, case_name, topological_perturbations), "gridopt-dataset-tmp"
    )


def _opf_group_dir(
    root: str,
    case_name: str,
    group_idx: int,
    topological_perturbations: bool,
) -> str:
    return os.path.join(
        _opf_tmp_dir(root, case_name, topological_perturbations),
        _opf_release_name(topological_perturbations),
        case_name,
        f"group_{group_idx}",
    )


def _opf_remote_url(case_name: str, group_idx: int, topological_perturbations: bool) -> str:
    release = _opf_release_name(topological_perturbations)
    return (
        "https://storage.googleapis.com/gridopt-dataset/"
        f"{release}/{case_name}_{group_idx}.tar.gz"
    )


def _is_address_family_not_supported(exc: Exception) -> bool:
    err = exc
    if isinstance(exc, urllib.error.URLError):
        err = getattr(exc, "reason", exc)
    return getattr(err, "errno", None) == 97


@contextmanager
def _force_ipv4_getaddrinfo():
    original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = _ipv4_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def _urlopen_with_ipv4_fallback(request_or_url, timeout: int):
    try:
        return urllib.request.urlopen(request_or_url, timeout=timeout)
    except urllib.error.URLError as exc:
        if not _is_address_family_not_supported(exc):
            raise
        logging.warning(
            "URL open hit Errno 97 (address family not supported); retrying with IPv4"
        )

    with _force_ipv4_getaddrinfo():
        return urllib.request.urlopen(request_or_url, timeout=timeout)


def _url_exists(url: str, timeout: int = 10) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with _urlopen_with_ipv4_fallback(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach {url}: {exc}") from exc


def _download_file(url: str, dst_path: str, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = f"{dst_path}.part.{os.getpid()}"
    try:
        with _urlopen_with_ipv4_fallback(url, timeout=timeout) as response:
            with open(tmp_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
        os.replace(tmp_path, dst_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _parallel_download_and_extract_opf(
    root,
    case_name,
    num_groups,
    topological_perturbations,
    rank,
    comm,
):
    raw_dir = _opf_raw_dir(root, case_name, topological_perturbations)
    os.makedirs(raw_dir, exist_ok=True)

    world_size = comm.Get_size()
    active_workers = min(world_size, num_groups)
    if rank >= active_workers:
        return

    assigned_groups = list(range(rank, num_groups, active_workers))
    logging.info(
        "OPF parallel fetch/extract: rank=%d assigned_groups=%d",
        rank,
        len(assigned_groups),
    )

    skipped_download = 0
    skipped_extract = 0
    downloaded = 0
    extracted = 0

    for group_idx in assigned_groups:
        archive_name = f"{case_name}_{group_idx}.tar.gz"
        archive_path = os.path.join(raw_dir, archive_name)
        group_dir = _opf_group_dir(
            root, case_name, group_idx, topological_perturbations
        )

        if not os.path.isfile(archive_path):
            archive_url = _opf_remote_url(
                case_name, group_idx, topological_perturbations
            )
            logging.info(
                "OPF parallel fetch: rank=%d case=%s group=%d action=download",
                rank,
                case_name,
                group_idx,
            )
            _download_file(archive_url, archive_path)
            downloaded += 1
        else:
            skipped_download += 1

        if os.path.isdir(group_dir):
            skipped_extract += 1
            continue

        logging.info(
            "OPF parallel extract: rank=%d case=%s group=%d action=extract",
            rank,
            case_name,
            group_idx,
        )
        tg_opf.extract_tar(archive_path, raw_dir)
        extracted += 1

    if skipped_download or skipped_extract:
        logging.info(
            "OPF parallel fetch/extract summary: rank=%d case=%s groups=%d downloaded=%d extracted=%d skip_download=%d skip_extract=%d",
            rank,
            case_name,
            len(assigned_groups),
            downloaded,
            extracted,
            skipped_download,
            skipped_extract,
        )


def _probe_remote_num_groups(
    case_name: str,
    topological_perturbations: bool,
    start_idx: int,
    max_groups: int,
) -> int:
    count = max(0, int(start_idx))
    for idx in range(count, max_groups):
        url = _opf_remote_url(case_name, idx, topological_perturbations)
        if _url_exists(url):
            count = idx + 1
            continue
        break
    return count


def _find_empty_json(root: str):
    empty = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) == 0:
                    empty.append(path)
            except OSError:
                empty.append(path)
    return empty


def _discover_cases(root: str, topological_perturbations: bool):
    release_dir = _opf_release_dir(root, topological_perturbations)
    if not os.path.isdir(release_dir):
        return []
    return sorted(
        name
        for name in os.listdir(release_dir)
        if os.path.isdir(os.path.join(release_dir, name))
    )


def _discover_num_groups(root: str, case_name: str, topological_perturbations: bool):
    raw_dir = _opf_raw_dir(root, case_name, topological_perturbations)
    if not os.path.isdir(raw_dir):
        return 0
    groups = []
    for name in os.listdir(raw_dir):
        if not name.startswith(f"{case_name}_") or not name.endswith(".tar.gz"):
            continue
        try:
            idx = int(name[len(case_name) + 1 : -len(".tar.gz")])
            groups.append(idx)
        except ValueError:
            continue
    return max(groups) + 1 if groups else 0


def _parse_num_groups(num_groups_arg: str) -> int | None:
    if isinstance(num_groups_arg, int):
        return num_groups_arg
    if str(num_groups_arg).lower() == "all":
        return None
    return int(num_groups_arg)


def _resolve_num_groups(
    requested_num_groups,
    datadir,
    case_name,
    topological_perturbations,
    num_groups_max,
    probe_remote,
    rank,
    comm,
):
    if requested_num_groups is not None:
        return requested_num_groups
    local_groups = _discover_num_groups(datadir, case_name, topological_perturbations)
    if not probe_remote:
        if local_groups == 0:
            if num_groups_max <= 0:
                raise RuntimeError(f"No groups found for case '{case_name}'.")
            return num_groups_max
        return local_groups
    if num_groups_max <= 0:
        raise RuntimeError(f"No groups found for case '{case_name}'.")
    if local_groups >= num_groups_max:
        return local_groups
    if rank == 0:
        logging.info(
            "Probing remote groups for %s starting at %d (cap %d)",
            case_name,
            local_groups,
            num_groups_max,
        )
        try:
            resolved = _probe_remote_num_groups(
                case_name,
                topological_perturbations,
                local_groups,
                num_groups_max,
            )
        except RuntimeError as exc:
            if local_groups > 0:
                logging.warning(
                    "Remote group probe failed for %s; using local groups=%d. "
                    "To disable remote probe, pass --no_num_groups_probe. Error: %s",
                    case_name,
                    local_groups,
                    exc,
                )
                resolved = local_groups
            else:
                raise
        if resolved == 0:
            raise RuntimeError(f"No groups found for case '{case_name}'.")
    else:
        resolved = None
    return comm.bcast(resolved, root=0)


def _reextract_opf_if_needed(root, case_name, num_groups, topological_perturbations):
    raw_dir = _opf_raw_dir(root, case_name, topological_perturbations)
    tmp_dir = _opf_tmp_dir(root, case_name, topological_perturbations)
    raw_files = [f"{case_name}_{i}.tar.gz" for i in range(num_groups)]

    if not os.path.isdir(raw_dir):
        logging.info(
            "OPF extract check: skip case=%s reason=raw_dir_missing path=%s",
            case_name,
            raw_dir,
        )
        return

    missing = [
        name for name in raw_files if not os.path.isfile(os.path.join(raw_dir, name))
    ]
    if missing:
        logging.info(
            "OPF extract check: skip case=%s reason=missing_archives count=%d",
            case_name,
            len(missing),
        )
        return

    if not os.path.isdir(tmp_dir):
        logging.info(
            "OPF extract check: skip case=%s reason=tmp_dir_missing (already processed)",
            case_name,
        )
        return

    empty = _find_empty_json(tmp_dir)
    if not empty:
        logging.info(
            "OPF extract check: skip case=%s reason=tmp_dir_healthy path=%s",
            case_name,
            tmp_dir,
        )
        return
    logging.warning(
        "OPF extract check: reextract case=%s reason=empty_json count=%d",
        case_name,
        len(empty),
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logging.info(
        "OPF extract check: extracting case=%s groups=%d",
        case_name,
        num_groups,
    )
    for name in raw_files:
        tg_opf.extract_tar(os.path.join(raw_dir, name), raw_dir)


def _ensure_missing_group_dirs(
    root,
    case_name,
    num_groups,
    topological_perturbations,
):
    raw_dir = _opf_raw_dir(root, case_name, topological_perturbations)
    missing_groups = []
    for group_idx in range(num_groups):
        group_dir = _opf_group_dir(root, case_name, group_idx, topological_perturbations)
        if not os.path.isdir(group_dir):
            missing_groups.append(group_idx)

    if not missing_groups:
        return

    logging.warning(
        "OPF extract check: missing group dirs case=%s count=%d; re-extracting archives",
        case_name,
        len(missing_groups),
    )
    for group_idx in missing_groups:
        archive_path = os.path.join(raw_dir, f"{case_name}_{group_idx}.tar.gz")
        if os.path.isfile(archive_path):
            tg_opf.extract_tar(archive_path, raw_dir)


def _ensure_opf_downloaded(
    root,
    case_name,
    num_groups,
    topological_perturbations,
    rank,
    comm,
):
    _parallel_download_and_extract_opf(
        root,
        case_name,
        num_groups,
        topological_perturbations,
        rank,
        comm,
    )
    comm.Barrier()

    if rank == 0:
        _reextract_opf_if_needed(
            root,
            case_name,
            num_groups,
            topological_perturbations,
        )
        _ensure_missing_group_dirs(
            root,
            case_name,
            num_groups,
            topological_perturbations,
        )
        logging.info(
            "OPF dataset ready: case=%s groups=%s topological_perturbations=%s",
            case_name,
            num_groups,
            topological_perturbations,
        )
    comm.Barrier()
