from dataclasses import dataclass

from download_and_uncompress_data import discover_cases, ensure_pglearn_downloaded


@dataclass(frozen=True)
class _DataOps:
    discover_cases: callable
    ensure_pglearn_downloaded: callable


data_ops = _DataOps(
    discover_cases=discover_cases,
    ensure_pglearn_downloaded=ensure_pglearn_downloaded,
)

__all__ = ["data_ops"]
