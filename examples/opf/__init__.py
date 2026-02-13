from dataclasses import dataclass

from download_and_uncompress_data import (
    _discover_cases,
    _ensure_opf_downloaded,
    _parse_num_groups,
    _resolve_num_groups,
)


@dataclass(frozen=True)
class _DataOps:
    discover_cases: callable
    ensure_opf_downloaded: callable
    parse_num_groups: callable
    resolve_num_groups: callable


data_ops = _DataOps(
    discover_cases=_discover_cases,
    ensure_opf_downloaded=_ensure_opf_downloaded,
    parse_num_groups=_parse_num_groups,
    resolve_num_groups=_resolve_num_groups,
)

__all__ = ["data_ops"]
