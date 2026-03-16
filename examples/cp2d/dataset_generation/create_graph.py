"""
Create PyTorch Geometric (PyG) graphs from Abaqus SVE *.inp files.

Assumptions (matching your notebook in `mesh.ipynb`):
- Grains are encoded as Abaqus element sets named like `GRAIN-<id>` and are exposed by
  meshio as `mesh.cell_sets['GRAIN-<id>']`.
- Nodes in the graph are grain centroids (projected to XY).
- Two grains are connected if any mesh node is incident to elements from both grains
  (i.e., they "touch" by sharing at least one FE node).

Outputs:
- A PyG `torch_geometric.data.Data` object with:
  - `pos`: (num_grains, 2) float tensor with XY centroids
  - `edge_index`: (2, num_edges*2) long tensor (undirected; both directions included)
  - `grain_ids`: (num_grains,) long tensor mapping graph node index -> original GRAIN id
  - `edge_attr`: (num_edges*2, 1) float tensor with centroid distance (optional)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import pandas as pd

try:
    import meshio  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "meshio is required to read Abaqus .inp files. Install with: `pip install meshio`."
    ) from e

from torch_geometric.data import Data


CentroidMode = Literal["unique_nodes", "node_occurrences"]
MaterialMissingPolicy = Literal["nan", "zero"]
YMissingPolicy = Literal["error", "nan"]


@dataclass(frozen=True)
class GrainExtraction:
    cell_block_idx: int
    cell_type: str
    grain_ids: np.ndarray  # (G,) sorted original grain ids
    grain_to_elem_indices: List[np.ndarray]  # len=G, each (n_elems_g,)
    elem_to_grain_node: np.ndarray  # (n_elems,) int64 in [0..G-1] or -1


def parse_graph_attr_from_inp_path(inp_path: str) -> Tuple[float, int, int, int]:
    """
    Extract graph-level attributes from the path/filename.

    Expected path contains a folder like `vf10` where vf means volume_fraction*100.
    Expected filename pattern:
      - SVE1_1x1_0.inp   -> (vf, sve_id=1, part=1, sub=0)
      - SVE21_2x2_1.inp  -> (vf, sve_id=21, part=2, sub=1)

    Returns: (volume_fraction, sve_id, partition_size, sub_partition_number)
    """
    # Volume fraction from folder segment 'vfXX'
    m_vf = re.search(r"(?:^|[\\/])vf(\d+)(?:[\\/]|$)", inp_path)
    if not m_vf:
        raise ValueError(f"Could not parse volume fraction from path (expected .../vfXX/...): {inp_path}")
    vf = int(m_vf.group(1)) / 100.0

    base = os.path.basename(inp_path)
    m = re.match(r"^SVE(\d+)_(\d+)x(\d+)_(\d+)\.inp$", base)
    if not m:
        raise ValueError(f"Could not parse SVE/partition/sub from filename: {base}")
    sve_id = int(m.group(1))
    part_x = int(m.group(2))
    part_y = int(m.group(3))
    sub = int(m.group(4))
    # You said the partition size is a single number; most of your files are square (NxN).
    # If a non-square ever appears, we still store part_x.
    _ = part_y

    return vf, sve_id, part_x, sub


def parse_vf_from_path(inp_path: str) -> Tuple[int, float]:
    """Return (vf_int, vf_float) from a path containing a `vfXX` segment."""
    m_vf = re.search(r"(?:^|[\\/])vf(\d+)(?:[\\/]|$)", inp_path)
    if not m_vf:
        raise ValueError(f"Could not parse volume fraction from path (expected .../vfXX/...): {inp_path}")
    vf_int = int(m_vf.group(1))
    return vf_int, vf_int / 100.0


def parse_sve_filename(inp_path: str) -> Tuple[int, int, int, int]:
    """
    Parse `SVE{num}_{px}x{py}_{sub}.inp` filename.
    Returns: (sve_id, part_x, part_y, sub)
    """
    base = os.path.basename(inp_path)
    m = re.match(r"^SVE(\d+)_(\d+)x(\d+)_(\d+)\.inp$", base)
    if not m:
        raise ValueError(f"Could not parse SVE/partition/sub from filename: {base}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def infer_grain_info_dir(inp_path: str) -> Optional[str]:
    """
    Infer `.../vfXX/vfXX_grain_info` from an inp path like `.../vfXX/2x2/SVE...inp`.
    Returns None if the directory doesn't exist.
    """
    m = re.search(r"^(.*?)(?:[\\/])vf(\d+)(?:[\\/])", inp_path)
    if not m:
        return None
    prefix, vf_digits = m.group(1), m.group(2)
    vf_dir = os.path.join(prefix, f"vf{vf_digits}")
    cand = os.path.join(vf_dir, f"vf{vf_digits}_grain_info")
    if os.path.isdir(cand):
        return cand
    return None


def infer_results_csv_path(inp_path: str, *, results_root: str) -> str:
    """
    Map an inp file to its graph-level results CSV:
      results/vf*/NxN/SVE#/sve#.csv
    """
    results_root = os.path.abspath(results_root)
    vf_int, _ = parse_vf_from_path(inp_path)
    sve_id, part_x, part_y, _sub = parse_sve_filename(inp_path)
    part_tag = f"{part_x}x{part_y}"
    return os.path.join(results_root, f"vf{vf_int}", part_tag, f"SVE{sve_id}", f"sve{sve_id}.csv")


def infer_results_csv_paths(inp_path: str, *, results_roots: Sequence[str]) -> List[str]:
    """Map an inp file to multiple results CSV paths (one per root)."""
    out: List[str] = []
    for root in results_roots:
        if not root:
            continue
        out.append(infer_results_csv_path(inp_path, results_root=root))
    return out


def infer_stress_strain_paths(
    inp_path: str, *, results_roots: Sequence[str], sub: int
) -> Tuple[Optional[str], Optional[str]]:
    """
    Map an inp file to stress-strain CSVs (x/y) stored under:
      results_root/vfXX/NxN/SVE#/stress_strain_csv/Domain_<sub>/SVE#_Domain<sub>_LC0_x.csv
      results_root/vfXX/NxN/SVE#/stress_strain_csv/Domain_<sub>/SVE#_Domain<sub>_LC1_y.csv
    Returns the first pair found across roots.
    """
    vf_int, _ = parse_vf_from_path(inp_path)
    sve_id, part_x, part_y, _ = parse_sve_filename(inp_path)
    part_tag = f"{part_x}x{part_y}"
    for root in results_roots:
        if not root:
            continue
        base = os.path.join(root, f"vf{vf_int}", part_tag, f"SVE{sve_id}", "stress_strain_csv")
        domain_dir = os.path.join(base, f"Domain_{sub}")
        x_path = os.path.join(domain_dir, f"SVE{sve_id}_Domain{sub}_LC0_x.csv")
        y_path = os.path.join(domain_dir, f"SVE{sve_id}_Domain{sub}_LC1_y.csv")
        if os.path.exists(x_path) and os.path.exists(y_path):
            return x_path, y_path
    return None, None


def load_graph_y_from_results_csv(
    csv_path: str,
    *,
    y_prefix: str = "out_",
    sve_id: Optional[int] = None,
    sub: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a results CSV (with header).

    Two supported layouts:
    1) "Row per subdomain" (what your 2x2/4x4/8x8 results look like):
       Columns include `inp_Index` and `inp_SubIndex`, with multiple rows.
       In this case we select the row for (inp_Index==sve_id, inp_SubIndex==sub+1).

    2) "Single row" (typical for 1x1): we select the last row.

    By default, selects columns starting with 'out_' as targets.
    Returns: (F,) float32
    """
    df = pd.read_csv(csv_path)
    if df.shape[0] < 1:
        raise ValueError(f"{csv_path} has no rows")

    # If subdomain info exists, prefer selecting by it.
    if "inp_SubIndex" in df.columns and sub is not None:
        # Convention in your CSVs: inp_SubIndex is 1-based, while sub in filename is 0-based.
        sub_idx = int(sub) + 1
        sel = df[df["inp_SubIndex"].astype(int) == sub_idx]
        if sve_id is not None and "inp_Index" in df.columns:
            sel = sel[sel["inp_Index"].astype(int) == int(sve_id)]
        if sel.shape[0] == 0:
            raise ValueError(
                f"{csv_path}: could not find row for inp_SubIndex={sub_idx}"
                + (f", inp_Index={sve_id}" if sve_id is not None else "")
            )
        row = sel.iloc[-1]
    else:
        row = df.iloc[-1]

    if y_prefix:
        cols = [c for c in df.columns if str(c).startswith(y_prefix)]
    else:
        cols = list(df.columns)
    if not cols:
        # fallback: take all numeric columns
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    y = row[cols].to_numpy(dtype=np.float32, copy=True)
    labels = [str(c) for c in cols]
    return y, labels


def load_stress_strain_curve(csv_path: str) -> np.ndarray:
    """
    Load a stress-strain curve CSV with header.
    Returns: (N, F) float32, with numeric columns preserved in file order.
    """
    df = pd.read_csv(csv_path)
    if df.shape[0] < 1:
        raise ValueError(f"{csv_path} has no rows")
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError(f"{csv_path} has no numeric columns")
    return df[cols].to_numpy(dtype=np.float32, copy=True)


def load_grain_info_material_features(
    csv_path: str,
    *,
    grain_ids: np.ndarray,
    material_missing: MaterialMissingPolicy = "nan",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a `SVE*_grain_info.csv` and align its feature rows to `grain_ids`.

    Expected input format (headerless), e.g.
      1,254.951,41.827,191.188,1.0,2.0,0.0
    where:
      - col 0 is grain_id
      - one column redundantly equals grain_id (often 5th column, i.e. col 4) -> drop it
      - last column is always 0.0 -> drop it

    Returns:
      - material: (G, F) float32
      - material_mask: (G,) bool, True when row existed & had no NaNs
    """
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} must have at least 2 columns (grain_id + features)")

    # Ensure grain_id int index
    df = df.rename(columns={0: "grain_id"})
    df["grain_id"] = df["grain_id"].astype(np.int64)

    # Detect and drop redundant "grain_id-like" columns (besides the true grain_id column).
    # We keep this detection tolerant of float formatting.
    grain_id_float = df["grain_id"].astype(np.float64)
    drop_cols: List[int] = []
    for c in df.columns:
        if c == "grain_id":
            continue
        # Only attempt numeric compare
        try:
            col = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        if col.isna().all():
            continue
        if np.allclose(col.to_numpy(dtype=np.float64), grain_id_float.to_numpy(), atol=1e-6, rtol=0.0):
            drop_cols.append(int(c))

    # Detect and drop all-zero column(s) (often the last column).
    for c in df.columns:
        if c == "grain_id":
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.isna().all():
            continue
        arr = col.to_numpy(dtype=np.float64)
        if np.allclose(arr, 0.0, atol=0.0, rtol=0.0):
            drop_cols.append(int(c))

    # Apply drops (ignore if duplicates)
    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.set_index("grain_id", drop=True)

    aligned = df.reindex(grain_ids)
    material_mask = ~aligned.isna().any(axis=1)
    material = aligned.to_numpy(dtype=np.float32, copy=False)
    if material_missing == "zero":
        material = np.nan_to_num(material, nan=0.0, posinf=0.0, neginf=0.0)

    return material, material_mask.to_numpy(dtype=bool, copy=False)


def _pick_cell_block(mesh: "meshio.Mesh") -> int:
    """
    Pick which mesh.cells[...] block to use.

    For your current SVE files, there is typically a single block (e.g. hexahedron).
    This chooses the first block that has any indices referenced by a GRAIN-* cell_set.
    """
    if not getattr(mesh, "cells", None):
        raise ValueError("mesh has no cells")
    if not getattr(mesh, "cell_sets", None):
        raise ValueError("mesh has no cell_sets (expected GRAIN-* element sets)")

    n_blocks = len(mesh.cells)
    # mesh.cell_sets[name] is a list aligned with cell blocks (sometimes length==n_blocks).
    # Some writers only emit the blocks that are relevant; handle both.
    for block_idx in range(n_blocks):
        for k, v in mesh.cell_sets.items():
            if not str(k).startswith("GRAIN-"):
                continue
            if not isinstance(v, list) or len(v) == 0:
                continue
            if len(v) == n_blocks:
                arr = v[block_idx]
            else:
                # single-block case (common here)
                arr = v[0]
            if arr is not None and len(arr) > 0:
                return block_idx

    # fallback: first block
    return 0


def extract_grains_from_mesh(mesh: "meshio.Mesh") -> GrainExtraction:
    """
    Extract grain element indices from `mesh.cell_sets['GRAIN-*']` and build an
    element->grain mapping for a single chosen cell block.
    """
    block_idx = _pick_cell_block(mesh)
    cells = mesh.cells[block_idx]
    n_elems = int(cells.data.shape[0])

    grain_map: Dict[int, np.ndarray] = {}
    n_blocks = len(mesh.cells)

    for key, blocks in getattr(mesh, "cell_sets", {}).items():
        k = str(key)
        if not k.startswith("GRAIN-"):
            continue
        try:
            grain_id = int(k.split("-", 1)[1])
        except Exception:
            continue

        if not isinstance(blocks, list) or len(blocks) == 0:
            continue
        if len(blocks) == n_blocks:
            indices = blocks[block_idx]
        else:
            indices = blocks[0]
        if indices is None or len(indices) == 0:
            continue

        grain_map[grain_id] = np.asarray(indices, dtype=np.int64)

    if not grain_map:
        raise ValueError("No grain element sets found. Expected cell_sets named 'GRAIN-<id>'.")

    grain_ids = np.array(sorted(grain_map.keys()), dtype=np.int64)
    grain_to_elem_indices = [grain_map[int(g)] for g in grain_ids]

    elem_to_grain_node = np.full((n_elems,), -1, dtype=np.int64)
    for node_idx, g in enumerate(grain_ids):
        elem_indices = grain_map[int(g)]
        elem_to_grain_node[elem_indices] = node_idx

    return GrainExtraction(
        cell_block_idx=block_idx,
        cell_type=cells.type,
        grain_ids=grain_ids,
        grain_to_elem_indices=grain_to_elem_indices,
        elem_to_grain_node=elem_to_grain_node,
    )


def compute_grain_centroids_xy(
    mesh: "meshio.Mesh",
    grains: GrainExtraction,
    *,
    centroid_mode: CentroidMode = "unique_nodes",
) -> np.ndarray:
    """
    Compute XY centroid for each grain.

    - unique_nodes: matches your notebook (mean of unique node coordinates in the grain)
    - node_occurrences: faster (mean over all node occurrences across the grain's elements)
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.shape[1] < 2:
        raise ValueError(f"mesh points are not at least 2D: {points.shape}")

    elems = np.asarray(mesh.cells[grains.cell_block_idx].data, dtype=np.int64)  # (E, k)

    pos = np.zeros((len(grains.grain_ids), 2), dtype=np.float64)

    if centroid_mode == "node_occurrences":
        # Fast approximation: average coordinates over all node occurrences.
        for i, elem_idx in enumerate(grains.grain_to_elem_indices):
            nids = elems[elem_idx].reshape(-1)
            pos[i] = points[nids, :2].mean(axis=0)
        return pos

    if centroid_mode != "unique_nodes":
        raise ValueError(f"Unknown centroid_mode={centroid_mode!r}")

    # Notebook-faithful: unique nodes per grain.
    for i, elem_idx in enumerate(grains.grain_to_elem_indices):
        nids = np.unique(elems[elem_idx].reshape(-1))
        pos[i] = points[nids, :2].mean(axis=0)
    return pos


def build_grain_adjacency_edge_index(
    mesh: "meshio.Mesh",
    grains: GrainExtraction,
    *,
    undirected: bool = True,
) -> np.ndarray:
    """
    Efficient grain adjacency from shared FE nodes using vectorized node grouping:
    - Flatten (element,node) connectivity to (node_id, grain_node_id) pairs
    - Group by node_id, connect all grains that appear in the same group
    """
    elems = np.asarray(mesh.cells[grains.cell_block_idx].data, dtype=np.int64)  # (E, k)
    elem_to_g = grains.elem_to_grain_node  # (E,)
    k = elems.shape[1]

    node_ids = elems.reshape(-1)
    grain_ids_flat = np.repeat(elem_to_g, k)
    mask = grain_ids_flat >= 0
    node_ids = node_ids[mask]
    grain_ids_flat = grain_ids_flat[mask]

    order = np.argsort(node_ids, kind="mergesort")
    node_sorted = node_ids[order]
    grain_sorted = grain_ids_flat[order]

    # segment boundaries where node id changes
    if len(node_sorted) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    boundaries = np.flatnonzero(node_sorted[1:] != node_sorted[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(node_sorted)]))

    edge_set: set[Tuple[int, int]] = set()
    for s, e in zip(starts, ends):
        g = np.unique(grain_sorted[s:e])
        if g.size < 2:
            continue
        # connect all grain pairs incident to this node
        for a, b in combinations(g.tolist(), 2):
            if a == b:
                continue
            if a < b:
                edge_set.add((a, b))
            else:
                edge_set.add((b, a))

    if not edge_set:
        return np.zeros((2, 0), dtype=np.int64)

    edges = np.array(sorted(edge_set), dtype=np.int64)  # (M,2) with a<b
    if undirected:
        u = np.concatenate([edges[:, 0], edges[:, 1]])
        v = np.concatenate([edges[:, 1], edges[:, 0]])
    else:
        u, v = edges[:, 0], edges[:, 1]

    return np.stack([u, v], axis=0)


def create_pyg_data_from_inp(
    inp_path: str,
    *,
    centroid_mode: CentroidMode = "unique_nodes",
    add_edge_attr_distance: bool = True,
    grain_info_dir: Optional[str] = None,
    material_missing: MaterialMissingPolicy = "nan",
    results_root: Optional[str] = None,
    results_root_extra: Optional[Sequence[str]] = None,
    y_prefix: str = "out_",
    y_missing: YMissingPolicy = "error",
) -> Data:
    mesh = meshio.read(inp_path)
    grains = extract_grains_from_mesh(mesh)
    pos = compute_grain_centroids_xy(mesh, grains, centroid_mode=centroid_mode)
    edge_index = build_grain_adjacency_edge_index(mesh, grains, undirected=True)

    data = Data(
        pos=torch.as_tensor(pos, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.int64),
        num_nodes=int(pos.shape[0]),
    )
    data.grain_ids = torch.as_tensor(grains.grain_ids, dtype=torch.int64)

    if add_edge_attr_distance and edge_index.shape[1] > 0:
        src = edge_index[0]
        dst = edge_index[1]
        d = torch.linalg.norm(data.pos[src] - data.pos[dst], dim=1, ord=2)
        data.edge_attr = d.view(-1, 1).to(torch.float32)

    # helpful metadata
    data.cell_type = grains.cell_type
    data.cell_block_idx = int(grains.cell_block_idx)

    # Graph-level attributes: [volume_fraction, sve_id, partition_size, sub_partition]
    vf, sve_id, part, sub = parse_graph_attr_from_inp_path(inp_path)
    data.graph_attr = torch.as_tensor([vf, float(sve_id), float(part), float(sub)], dtype=torch.float32)

    # Graph-level output y from results CSV(s)
    results_roots: List[str] = []
    if results_root:
        results_roots.append(results_root)
    if results_root_extra:
        results_roots.extend([r for r in results_root_extra if r])
    if results_roots:
        y_paths = infer_results_csv_paths(inp_path, results_roots=results_roots)
        y_parts: List[np.ndarray] = []
        y_labels_parts: List[List[str]] = []
        missing_paths: List[str] = []
        # Use sve_id/sub from the filename so we pick the correct row when results CSV has multiple rows.
        sve_id2, _px, _py, sub2 = parse_sve_filename(inp_path)
        for y_path in y_paths:
            if os.path.exists(y_path):
                y_part, labels_part = load_graph_y_from_results_csv(
                    y_path, y_prefix=y_prefix, sve_id=sve_id2, sub=sub2
                )
                y_parts.append(y_part)
                y_labels_parts.append(labels_part)
            else:
                missing_paths.append(y_path)
        if missing_paths and y_missing == "error":
            missing_str = "\n".join(missing_paths)
            raise FileNotFoundError(f"Missing results file(s) for {inp_path}:\n{missing_str}")
        if y_parts:
            y = np.concatenate(y_parts, axis=0)
            data.y = torch.as_tensor(y, dtype=torch.float32)
            data.y_labels = [c for part in y_labels_parts for c in part]
        else:
            # nan fill: unknown number of targets; we can't infer without a file.
            data.y = torch.full((0,), float("nan"), dtype=torch.float32)

        # Optional stress-strain curves (x/y) from csv_results3-style folders.
        x_path, y_path = infer_stress_strain_paths(inp_path, results_roots=results_roots, sub=sub2)
        if x_path and y_path:
            x_curve = load_stress_strain_curve(x_path)
            y_curve = load_stress_strain_curve(y_path)
            data.stress_strain_x = torch.as_tensor(x_curve, dtype=torch.float32)
            data.stress_strain_y = torch.as_tensor(y_curve, dtype=torch.float32)

    # Add volfrac from y into graph_attr when available.
    if hasattr(data, "y_labels") and hasattr(data, "y") and data.y.numel() > 0:
        label_to_idx = {str(c): i for i, c in enumerate(data.y_labels)}
        vol_idx = None
        for key in ("volfrac", "inp_VolFrac"):
            if key in label_to_idx:
                vol_idx = label_to_idx[key]
                break
        if vol_idx is not None:
            volfrac = data.y[int(vol_idx)].view(1)
        else:
            volfrac = torch.as_tensor([float("nan")], dtype=torch.float32)
        data.graph_attr = torch.cat([data.graph_attr, volfrac.to(torch.float32)], dim=0)

    # Attach per-grain "material" info (CSV is keyed by grain_id in col 0)
    if grain_info_dir:
        grain_info_dir = os.path.abspath(grain_info_dir)
        m = re.search(r"SVE(\d+)", os.path.basename(inp_path))
        if m:
            sve_id = int(m.group(1))
            csv_path = os.path.join(grain_info_dir, f"SVE{sve_id}_grain_info.csv")
            if os.path.exists(csv_path):
                mat, mask = load_grain_info_material_features(
                    csv_path,
                    grain_ids=grains.grain_ids,
                    material_missing=material_missing,
                )

                # Add number of elements in the grain:
                # - as float in the last column of `data.material` (for convenience in ML pipelines)
                # - as int64 in `data.n_elems` (for exact integer usage)
                n_elems_i64 = np.asarray([len(e) for e in grains.grain_to_elem_indices], dtype=np.int64)
                n_elems_f32 = n_elems_i64.astype(np.float32).reshape(-1, 1)
                mat = np.concatenate([mat, n_elems_f32], axis=1)

                data.material = torch.as_tensor(mat, dtype=torch.float32)
                data.n_elems = torch.as_tensor(n_elems_i64, dtype=torch.int64)
                data.material_mask = torch.as_tensor(mask, dtype=torch.bool)
                data.grain_info_csv = os.path.basename(csv_path)
            else:
                # keep graph usable even if CSV missing
                data.grain_info_csv = None
        else:
            data.grain_info_csv = None
    return data


def data_to_safe_dict(data: Data) -> Dict[str, torch.Tensor]:
    """
    Convert a PyG Data object into a plain-tensor dict that is safe to `torch.save`/`torch.load`
    under PyTorch's `weights_only=True` default (PyTorch>=2.6).
    """
    out: Dict[str, torch.Tensor] = {
        "pos": data.pos,
        "edge_index": data.edge_index,
        "grain_ids": data.grain_ids,
    }
    if hasattr(data, "graph_attr"):
        out["graph_attr"] = data.graph_attr
    if hasattr(data, "y"):
        out["y"] = data.y
    if hasattr(data, "edge_attr"):
        out["edge_attr"] = data.edge_attr
    if hasattr(data, "material"):
        out["material"] = data.material
    if hasattr(data, "n_elems"):
        out["n_elems"] = data.n_elems
    if hasattr(data, "material_mask"):
        out["material_mask"] = data.material_mask
    if hasattr(data, "stress_strain_x"):
        out["stress_strain_x"] = data.stress_strain_x
    if hasattr(data, "stress_strain_y"):
        out["stress_strain_y"] = data.stress_strain_y
    # store small metadata as tensors for simplicity/portability
    out["cell_block_idx"] = torch.as_tensor([int(getattr(data, "cell_block_idx", 0))], dtype=torch.int64)
    # cell_type is a string; don't force it into tensors (keep out of safe dict)
    return out


def safe_dict_to_data(d: Dict[str, torch.Tensor]) -> Data:
    """Reconstruct a PyG Data object from `data_to_safe_dict` output."""
    data = Data(
        pos=d["pos"],
        edge_index=d["edge_index"],
        num_nodes=int(d["pos"].shape[0]),
    )
    data.grain_ids = d["grain_ids"]
    if "graph_attr" in d:
        data.graph_attr = d["graph_attr"]
    if "y" in d:
        data.y = d["y"]
    if "edge_attr" in d:
        data.edge_attr = d["edge_attr"]
    if "material" in d:
        data.material = d["material"]
    if "n_elems" in d:
        data.n_elems = d["n_elems"]
    if "material_mask" in d:
        data.material_mask = d["material_mask"]
    if "stress_strain_x" in d:
        data.stress_strain_x = d["stress_strain_x"]
    if "stress_strain_y" in d:
        data.stress_strain_y = d["stress_strain_y"]
    if "cell_block_idx" in d:
        data.cell_block_idx = int(d["cell_block_idx"].view(-1)[0].item())
    return data


def _expand_inp_paths(inp: Optional[str], inp_glob: Optional[str]) -> List[str]:
    paths: List[str] = []
    if inp:
        paths.append(inp)
    if inp_glob:
        paths.extend(sorted(glob.glob(inp_glob)))
    # de-dupe while preserving order
    seen = set()
    out = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            out.append(ap)
            seen.add(ap)
    return out


def _parse_results_roots_arg(roots_arg: Optional[str]) -> List[str]:
    if not roots_arg:
        return []
    return [os.path.abspath(s) for s in roots_arg.split(",") if s]


def _resolve_out_dir(p: str, args: argparse.Namespace) -> str:
    if args.out_dir:
        return args.out_dir
    if args.inp_root:
        return os.path.join(os.path.abspath(args.inp_root), "graphs")
    return os.path.dirname(p)


def discover_inp_paths(
    inp_root: str,
    *,
    vf_folders: Sequence[str],
    partition_folders: Sequence[str],
) -> List[str]:
    """
    Discover SVE*.inp files under:
      <inp_root>/<vf_folder>/<partition_folder>/SVE*.inp
    """
    inp_root = os.path.abspath(inp_root)
    out: List[str] = []
    for vf in vf_folders:
        for part in partition_folders:
            pattern = os.path.join(inp_root, vf, part, "SVE*.inp")
            out.extend(sorted(glob.glob(pattern)))
    # de-dupe while preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            uniq.append(ap)
            seen.add(ap)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=str, default=None, help="Path to a single SVE .inp file")
    ap.add_argument("--inp_glob", type=str, default=None, help="Glob for multiple .inp files (e.g. 'SVE*.inp')")
    ap.add_argument(
        "--inp_root",
        type=str,
        default=None,
        help="Root directory containing vf folders (e.g. '/.../inp_files'). If set and --inp/--inp_glob are omitted, runs in batch mode.",
    )
    ap.add_argument(
        "--vf_folders",
        type=str,
        default="vf10,vf45,vf90",
        help="Comma-separated vf folders to process in batch mode (default: vf10,vf45,vf90).",
    )
    ap.add_argument(
        "--partition_folders",
        type=str,
        default="1x1,2x2,4x4,8x8",
        help="Comma-separated partition subfolders to process in batch mode (default: 1x1,2x2,4x4,8x8).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for .pt files (default: <inp_root>/graphs in batch mode, else input dir).",
    )
    ap.add_argument(
        "--grain_info_dir",
        type=str,
        default="auto",
        help="Directory containing SVE*_grain_info.csv files. Use 'auto' to infer vf##_grain_info per file, or provide an explicit path.",
    )
    ap.add_argument(
        "--results_root",
        type=str,
        default="",
        help="Primary results directory containing vf*/NxN/SVE#/sve#.csv. Set '' to disable.",
    )
    ap.add_argument(
        "--results_root_extra",
        type=str,
        default="",
        help="Additional result roots with the same structure. Comma-separated list; set '' to disable.",
    )
    ap.add_argument(
        "--y_prefix",
        type=str,
        default="",
        help="Which columns to take as targets from results CSV (prefix match). Use '' to take all numeric columns.",
    )
    ap.add_argument(
        "--y_missing",
        type=str,
        default="error",
        choices=["error", "nan"],
        help="What to do if results CSV is missing for a file.",
    )
    ap.add_argument(
        "--y_labels_csv",
        type=str,
        default=None,
        help="Optional path to write one-line CSV of y labels. Default: <out_dir>/y_labels.csv.",
    )
    ap.add_argument(
        "--material_missing",
        type=str,
        default="nan",
        choices=["nan", "zero"],
        help="How to fill missing grain_id rows in the grain_info CSV.",
    )
    ap.add_argument(
        "--centroid_mode",
        type=str,
        default="unique_nodes",
        choices=["unique_nodes", "node_occurrences"],
        help="How to compute grain centroids (unique_nodes matches the notebook; node_occurrences is faster).",
    )
    ap.add_argument(
        "--no_edge_attr_distance",
        action="store_true",
        help="Disable edge_attr = centroid distance.",
    )
    ap.add_argument(
        "--save_format",
        type=str,
        default="dict",
        choices=["dict", "data"],
        help="Saving format: 'dict' is torch.load-safe (recommended); 'data' pickles PyG Data.",
    )
    args = ap.parse_args()

    inp_paths = _expand_inp_paths(args.inp, args.inp_glob)
    if not inp_paths and args.inp_root:
        vf_folders = [s for s in (args.vf_folders or "").split(",") if s]
        part_folders = [s for s in (args.partition_folders or "").split(",") if s]
        inp_paths = discover_inp_paths(args.inp_root, vf_folders=vf_folders, partition_folders=part_folders)
    if not inp_paths:
        raise SystemExit("No input files. Use --inp, --inp_glob, or --inp_root (batch mode).")

    results_root_extra = _parse_results_roots_arg(args.results_root_extra)
    y_labels_ref: Optional[List[str]] = None
    labels_out_dir: Optional[str] = None

    for p in inp_paths:
        # per-file grain-info directory
        if args.grain_info_dir == "auto":
            grain_info_dir = infer_grain_info_dir(p)
        else:
            grain_info_dir = args.grain_info_dir
        if grain_info_dir == "":
            grain_info_dir = None

        data = create_pyg_data_from_inp(
            p,
            centroid_mode=args.centroid_mode,
            add_edge_attr_distance=not args.no_edge_attr_distance,
            grain_info_dir=grain_info_dir,
            material_missing=args.material_missing,
            results_root=args.results_root,
            results_root_extra=results_root_extra,
            y_prefix=args.y_prefix,
            y_missing=args.y_missing,
        )

        out_dir = _resolve_out_dir(p, args)
        os.makedirs(out_dir, exist_ok=True)

        # Output filename includes vf# and partition size (as requested)
        vf_int, _ = parse_vf_from_path(p)
        sve_id, part_x, part_y, sub = parse_sve_filename(p)
        part_tag = f"{part_x}x{part_y}"
        base = os.path.splitext(os.path.basename(p))[0]
        out_name = f"vf{vf_int}_{part_tag}_{base}.pt"
        out_path = os.path.join(out_dir, out_name)
        if args.save_format == "data":
            torch.save(data, out_path)
        else:
            torch.save(data_to_safe_dict(data), out_path)
        undirected_edges = int(data.edge_index.shape[1] // 2)
        print(
            f"[OK] {os.path.basename(p)} -> {out_path} | grains={data.num_nodes} "
            f"edges={undirected_edges} save_format={args.save_format}"
        )

        if hasattr(data, "y_labels"):
            if y_labels_ref is None:
                y_labels_ref = list(data.y_labels)
            elif list(data.y_labels) != y_labels_ref:
                print("[WARN] y_labels differ across files; writing the first set.")
            if labels_out_dir is None:
                labels_out_dir = out_dir
            elif labels_out_dir != out_dir:
                print(f"[WARN] Multiple output dirs; writing y_labels.csv to {labels_out_dir}")

    if y_labels_ref:
        if args.y_labels_csv == "":
            return
        if args.y_labels_csv:
            labels_path = args.y_labels_csv
        else:
            labels_out_dir = labels_out_dir or os.getcwd()
            labels_path = os.path.join(labels_out_dir, "y_labels.csv")
        labels_dir = os.path.dirname(labels_path)
        if labels_dir:
            os.makedirs(labels_dir, exist_ok=True)
        with open(labels_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(y_labels_ref)
        print(f"[OK] y_labels -> {labels_path}")


if __name__ == "__main__":
    main()

