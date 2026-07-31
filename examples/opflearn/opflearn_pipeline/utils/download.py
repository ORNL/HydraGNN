
import logging
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

ARCHIVE_URL = "https://data.nlr.gov/system/files/177/OPFLearn_Datasets.zip"
CASE_URL_TEMPLATE = "https://data.nlr.gov/system/files/177/pglib_opf_{case}.csv"
DEFAULT_CASES = [
    "case5_pjm",
    "case14_ieee",
    "case30_ieee",
    "case57_ieee",
    "case118_ieee",
]


def create_session() -> requests.Session:
    """Create a requests session with retry policy for transient failures."""
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": "opflearn-pipeline/1.0"})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def human_bytes(n_bytes: int) -> str:
    """Format file size in a human-readable way."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n_bytes} B"


def should_skip(path: Path, force: bool) -> bool:
    """Return True when an existing non-empty file should not be re-downloaded."""
    return path.exists() and path.is_file() and path.stat().st_size > 0 and not force


def stream_download(
    url: str,
    output_path: Path,
    force: bool,
    session: requests.Session,
    logger: logging.Logger,
) -> Path:
    """Download a URL atomically via ``.part`` file with streaming and retries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if should_skip(output_path, force=force):
        logger.info("Skipping existing file: %s (%s)", output_path, human_bytes(output_path.stat().st_size))
        return output_path

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    logger.info("Downloading %s", url)
    with session.get(url, stream=True, timeout=120, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0"))
        with open(part_path, "wb") as handle:
            progress = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
            )
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
            progress.close()

    part_size = part_path.stat().st_size
    if part_size <= 0:
        raise RuntimeError(f"Downloaded file is empty: {url}")

    part_path.replace(output_path)
    logger.info("Downloaded %s (%s)", output_path, human_bytes(output_path.stat().st_size))
    return output_path


def extract_archive(archive_path: Path, extract_dir: Path, logger: logging.Logger) -> list[Path]:
    """Extract OPFLearn archive and return discovered CSV files."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s", archive_path, extract_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    csv_files = sorted(extract_dir.rglob("*.csv"))
    logger.info("Found %d CSV files after extraction.", len(csv_files))
    for csv_file in csv_files:
        logger.info("CSV: %s", csv_file)
    return csv_files


def download_archive(raw_dir: Path, force: bool, logger: logging.Logger) -> Path:
    """Download the full OPFLearn ZIP archive."""
    session = create_session()
    try:
        target = raw_dir / "OPFLearn_Datasets.zip"
        return stream_download(ARCHIVE_URL, target, force=force, session=session, logger=logger)
    finally:
        session.close()


def download_cases(cases: list[str], raw_dir: Path, force: bool, logger: logging.Logger) -> list[Path]:
    """Download selected OPFLearn case CSV files."""
    session = create_session()
    downloaded: list[Path] = []
    try:
        for case in cases:
            case_name = case.strip()
            if not case_name:
                continue
            url = CASE_URL_TEMPLATE.format(case=case_name)
            output = raw_dir / f"pglib_opf_{case_name}.csv"
            downloaded.append(stream_download(url, output, force=force, session=session, logger=logger))
    finally:
        session.close()
    return downloaded
