from __future__ import annotations

import glob
import json
import os
import os.path as osp
import pickle
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

import hydragnn

from hydragnn.utils.datasets.pickledataset import SimplePickleDataset, SimplePickleWriter

try:
    from .cp_helpers import (
        EulerAngleMismatch,
        PhasePairCode,
        QuaternionAngleMismatch,
        QuaternionRelativeRotation,
        euler_bunge_zxz_to_quat,
    )
except ImportError:  # pragma: no cover
    from cp_helpers import (
        EulerAngleMismatch,
        PhasePairCode,
        QuaternionAngleMismatch,
        QuaternionRelativeRotation,
        euler_bunge_zxz_to_quat,
    )

try:
    from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
    from sklearn.preprocessing import StandardScaler as SkStandardScaler
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "scikit-learn is required for CP2D normalization. Install with: `pip install scikit-learn`."
    ) from e

ScalerType = Literal["none", "standard", "minmax"]
YColsMode = Literal["numeric", "out_prefix"]
EdgeAttrMode = Literal["none", "distance"]
ExtraEdgeAttr = Literal["euler_mismatch", "quat_angle", "quat_rel", "phase_pair"]
XMode = Literal["euler", "quat"]

def infer_y_labels(results_csv: str, *, mode: YColsMode = "numeric") -> List[str]:
    """
    Infer y-label order from a representative results CSV header.

    - numeric: all numeric columns (matches your current `create_graph.py` after the edit)
    - out_prefix: only columns starting with 'out_' (fallbacks to numeric if none found)
    """
    df = pd.read_csv(results_csv, nrows=1)
    if mode == "out_prefix":
        cols = [c for c in df.columns if str(c).startswith("out_")]
        if cols:
            return cols
    return [c for c in df.columns]


def _safe_dict_to_data(d: Dict[str, torch.Tensor]) -> Data:
    """Reconstruct minimal PyG Data from the dict saved by `create_graph.py --save_format dict`."""
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


def _compute_edge_distance(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    return torch.linalg.norm(pos[col] - pos[row], dim=1, ord=2).view(-1, 1)


def resolve_y_labels_csv(graphs_dir: str, y_labels_csv: Optional[str]) -> str:
    """
    Resolve the header-only CSV emitted by `create_graph.py`.

    By default, `create_graph.py` writes `y_labels.csv` into the graph output folder,
    so the published repo infers it from `graphs_dir` when no explicit path is given.
    """
    if y_labels_csv:
        resolved = osp.abspath(y_labels_csv)
        if not osp.exists(resolved):
            raise FileNotFoundError(f"y_labels_csv not found: {resolved}")
        return resolved

    candidate = osp.join(osp.abspath(graphs_dir), "y_labels.csv")
    if osp.exists(candidate):
        return candidate

    raise FileNotFoundError(
        "Could not infer `y_labels.csv`. Pass `y_labels_csv=...` explicitly or "
        "place the file in the graph directory produced by `create_graph.py`."
    )


class CP2D(SimplePickleDataset):
    """
    CP2D dataset, modeled after `CPSAGE` usage in `train_pre copy.py`.

    Source: graph `.pt` files saved by root `create_graph.py` (safe dict format).
    Output: HydraGNN pickled dataset via `SimplePickleWriter`, then loadable via `SimplePickleDataset`.

    Returned Data (simplified):
    - pos
    - x: selected from `material` using labels: phi1, Phi, phi2, phase, size
    - y: concatenated stress-strain curves (x then y) from graph fields
    - edge_index
    - edge_attr: optional (distance and/or cp_helpers augmentations)
    """

    X_LABELS_EULER: List[str] = ["phi1", "Phi", "phi2", "phase", "size"]
    X_LABELS_QUAT: List[str] = ["q0", "q1", "q2", "q3", "phase", "size"]
    # Graph-level scalar appended to every node as an extra node feature column.
    # Sourced from `graph_attr[4]` in the raw graphs.
    X_LABEL_VF_TRUE: str = "vf_true"
    # Vector outputs: stress-strain curves (stress values only) for x and y directions.
    Y_LABELS_VECTOR: List[str] = ["stress_strain_x", "stress_strain_y"]
    DERIVED_Y_LABELS: List[str] = ["K_R", "G_R", "out_AU", "n_avg", "K_avg"]
    STRESS_STRAIN_LEN: int = 24
    # Use 0-based row indices from the raw stress-strain curve.
    # STRESS_STRAIN_ROWS_0B: Tuple[int, ...] = (2, 5, 8, 10, 12, 14, 16, 18, 20, 22)
    STRESS_STRAIN_ROWS_0B: Tuple[int, ...] = tuple(range(24))

    def __init__(
        self,
        root: str,
        var_config,
        force_reload: bool = False,
        scaler: ScalerType = "none",
        *,
        # node feature representation
        x_mode: XMode = "euler",
        # source graphs
        graphs_dir: Optional[str] = None,
        graphs_glob: str = "*.pt",
        # selection
        x_labels: Optional[Sequence[str]] = None,
        y_labels: Optional[Sequence[str]] = None,
        y_labels_csv: Optional[str] = None,
        y_cols_mode: YColsMode = "numeric",
        # edge attrs
        edge_attr_mode: EdgeAttrMode = "distance",
        extra_edge_attrs: Optional[Sequence[ExtraEdgeAttr]] = None,
        # reproducibility
        shuffle: bool = True,
        shuffle_seed: int = 42,
    ):
        self.root = root
        self.var_config = var_config
        self.label = "data"
        self.scaler = scaler

        self.graphs_dir = osp.abspath(graphs_dir or osp.join(self.root, "graphs"))
        self.graphs_glob = graphs_glob

        self.x_mode: XMode = x_mode
        self.x_labels_all = (
            list(self.X_LABELS_EULER) if self.x_mode == "euler" else list(self.X_LABELS_QUAT)
        )
        if self.X_LABEL_VF_TRUE not in self.x_labels_all:
            self.x_labels_all.append(self.X_LABEL_VF_TRUE)
        self.x_labels = list(x_labels) if x_labels is not None else list(self.x_labels_all)
        self.y_labels_csv = resolve_y_labels_csv(self.graphs_dir, y_labels_csv)
        # Vector outputs + scalar outputs from graph `y`.
        self.y_labels_raw = infer_y_labels(self.y_labels_csv, mode=y_cols_mode)
        self.y_labels_scalar_all = list(self.y_labels_raw)
        for lab in self.DERIVED_Y_LABELS:
            if lab not in self.y_labels_scalar_all:
                self.y_labels_scalar_all.append(lab)
        self.y_labels_all = list(self.Y_LABELS_VECTOR) + [
            lab for lab in self.y_labels_scalar_all if lab not in self.Y_LABELS_VECTOR
        ]
        self.y_labels = list(y_labels) if y_labels is not None else list(self.y_labels_all)

        self.edge_attr_mode = edge_attr_mode
        self.extra_edge_attrs = list(extra_edge_attrs) if extra_edge_attrs is not None else []

        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)

        self.pickle_basedir = osp.join(self.processed_dir, f"{self.label}.pickle")
        self.meta_path = osp.join(self.pickle_basedir, f"{self.label}-meta.pkl")

        self.var_config["input_node_features"] = list(range(len(self.x_labels)))
        self.var_config["node_feature_names"] = list(self.x_labels)
        self.var_config["node_feature_dims"] = [1] * len(self.x_labels)

        if force_reload or not osp.exists(self.meta_path):
            self.process()

        super().__init__(basedir=self.pickle_basedir, label=self.label, var_config=self.var_config)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")

    @staticmethod
    def _labels_to_indices(labels: Sequence[str], universe: Sequence[str], *, name: str) -> List[int]:
        u = {k: i for i, k in enumerate(universe)}
        out: List[int] = []
        missing: List[str] = []
        for lab in labels:
            if lab not in u:
                missing.append(lab)
            else:
                out.append(u[lab])
        if missing:
            raise ValueError(f"Unknown {name}: {missing}. Available: {list(universe)}")
        return out

    @staticmethod
    def _make_scaler(mode: ScalerType):
        if mode == "standard":
            return SkStandardScaler(with_mean=True, with_std=True)
        if mode == "minmax":
            return SkMinMaxScaler()
        return None

    @staticmethod
    def _augment_y_with_derived(y_raw: torch.Tensor, y_labels_raw: Sequence[str]):
        """
        Append derived CP2D targets to the end of the raw y-vector.

        Derived targets (2D plane strain, 3x3 Voigt ordering [11, 22, 12]) are computed
        from the **compliance** tensor D = C^{-1} using Reuss-style isotropization formulas:

        - K_R:
            K2D = 1 / (D11 + D22 + 2*D12)
        - G_R (Reuss):
            D_iso33 = 1/2*(D11 + D22) - D12 + 1/2*D33
            mu_R = 1 / D_iso33

        Indices above are 1-based; in code they map to D[0,0], D[1,1], D[0,1], D[2,2].

        Notes:
        - Uses 2D Voigt ordering: [11, 22, 12] with engineering shear strain γ12 = 2 ε12.
        - Handles (small) coupling terms C13,C23,C31,C32 via the full 3x3 inversion.
        """
        y = y_raw.to(torch.float32).view(-1)
        labels = list(y_labels_raw)
        idx = {k: i for i, k in enumerate(labels)}

        def _get(name: str) -> torch.Tensor:
            if name not in idx:
                raise ValueError(
                    f"Cannot compute derived target because '{name}' is missing from y_labels_raw. "
                    f"Available (raw) labels include: {labels}"
                )
            return y[idx[name]]

        # Build 2D (plane) stiffness in Voigt form.
        c11 = _get("out_C11")
        c12 = _get("out_C12")
        c13 = _get("out_C13")
        c21 = _get("out_C21")
        c22 = _get("out_C22")
        c23 = _get("out_C23")
        c31 = _get("out_C31")
        c32 = _get("out_C32")
        c33 = _get("out_C33")

        C0 = torch.stack(
            [
                torch.stack([c11, c12, c13]),
                torch.stack([c21, c22, c23]),
                torch.stack([c31, c32, c33]),
            ],
            dim=0,
        )
        # Enforce symmetry (numerical noise / minor asymmetry).
        C0 = 0.5 * (C0 + C0.T)

        eps_reg = 1e-12
        eye3 = torch.eye(3, dtype=C0.dtype, device=C0.device)
        D0 = torch.linalg.inv(C0 + eps_reg * eye3)

        # K_R
        K_R = 1.0 / (D0[0, 0] + D0[1, 1] + 2.0 * D0[0, 1])

        # G_R (Reuss), per your formula:
        # D_iso33 = 1/2*(D11 + D22) - D12 + 1/2*D33
        D_iso33 = 0.5 * (D0[0, 0] + D0[1, 1]) - D0[0, 1] + 0.5 * D0[2, 2]
        G_R = 1.0 / D_iso33

        # Ranganathan–Ostoja-Starzewski "universal" anisotropy index (2D analog).
        # In 3D: A^U = 5*(G_V/G_R) + (K_V/K_R) - 6 (Ranganathan & Ostoja-Starzewski, 2008).
        # For a 2D (in-plane) elastic system, deviatoric strain space has dimension 2 (vs 5 in 3D),
        # so the consistent 2D analog is:
        #   A^U_2D = 2*(G_V/G_R) + (K_V/K_R) - 3
        #
        # K_V (uniform strain, 2D): (C11 + C22 + 2*C12)/4
        # G_V (uniform strain, 2D) computed by averaging two independent unit deviatoric modes:
        #   (ε11,ε22,γ12) = (1/2,-1/2,0) and (0,0,1) in engineering-shear convention.
        K_V = 0.25 * (C0[0, 0] + C0[1, 1] + 2.0 * C0[0, 1])
        G_V = 0.125 * (C0[0, 0] + C0[1, 1] - 2.0 * C0[0, 1]) + 0.5 * C0[2, 2]
        out_AU = 2.0 * (G_V / (G_R + eps_reg)) + (K_V / (K_R + eps_reg)) - 3.0

        # Direction-averaged hardening parameters (when both x/y components exist).
        # These are often used as scalar summaries of (nx, ny) and (Kx, Ky).
        n_avg = None
        K_avg = None
        if "nx" in idx and "ny" in idx:
            n_avg = 0.5 * (_get("nx") + _get("ny"))
        if "Kx" in idx and "Ky" in idx:
            K_avg = 0.5 * (_get("Kx") + _get("Ky"))

        extras: List[torch.Tensor] = []
        extra_labels: List[str] = []
        if "K_R" not in idx:
            extras.append(K_R)
            extra_labels.append("K_R")
        if "G_R" not in idx:
            extras.append(G_R)
            extra_labels.append("G_R")
        if "out_AU" not in idx:
            extras.append(out_AU)
            extra_labels.append("out_AU")
        if n_avg is not None and "n_avg" not in idx:
            extras.append(n_avg)
            extra_labels.append("n_avg")
        if K_avg is not None and "K_avg" not in idx:
            extras.append(K_avg)
            extra_labels.append("K_avg")

        if extras:
            y = torch.cat([y, torch.stack(extras).view(-1)], dim=0)
            labels.extend(extra_labels)

        return y, labels

    def process(self) -> None:
        os.makedirs(self.pickle_basedir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        graph_paths = sorted(glob.glob(osp.join(self.graphs_dir, self.graphs_glob)))
        if not graph_paths:
            raise FileNotFoundError(f"No graph .pt files found in {self.graphs_dir} matching {self.graphs_glob}")

        if self.shuffle:
            rng = np.random.default_rng(self.shuffle_seed)
            perm = rng.permutation(len(graph_paths)).tolist()
            graph_paths = [graph_paths[i] for i in perm]
            with open(osp.join(self.processed_dir, "shuffle.json"), "w") as f:
                json.dump({"seed": self.shuffle_seed, "perm": perm, "n": len(perm)}, f)

        x_idx = self._labels_to_indices(self.x_labels, self.x_labels_all, name="x_labels")

        # scalers (column-wise)
        sx = self._make_scaler(self.scaler)
        sy = self._make_scaler(self.scaler)
        spos = self._make_scaler(self.scaler)
        sedge = self._make_scaler(self.scaler)

        # Fit scalers if needed
        if self.scaler != "none":
            for p in tqdm(graph_paths, desc="Fit scalers"):
                d = torch.load(p)  # safe dict
                g = _safe_dict_to_data(d)
                pos = g.pos.to(torch.float32)
                material = g.material.to(torch.float32)
                euler = material[:, 0:3]  # always available from material
                if self.x_mode == "euler":
                    x_all = material
                elif self.x_mode == "quat":
                    quat = euler_bunge_zxz_to_quat(euler, degrees=True).to(torch.float32)
                    phase_size = material[:, 3:5]
                    x_all = torch.cat([quat, phase_size], dim=1)
                else:
                    raise ValueError(f"Unknown x_mode={self.x_mode}")

                # Append per-graph scalar (`graph_attr[4]`) as a constant extra node feature column.
                if not hasattr(g, "graph_attr"):
                    raise ValueError("Graph is missing `graph_attr`; cannot append vf_true to node features.")
                if len(g.graph_attr) <= 4:
                    raise ValueError(
                        f"`graph_attr` is too short (len={len(g.graph_attr)}); expected `graph_attr[4]` for vf_true."
                    )
                vf_true = _as_float(g.graph_attr[4])
                vf_true_col = torch.full(
                    (x_all.shape[0], 1),
                    vf_true,
                    dtype=torch.float32,
                    device=x_all.device,
                )
                x_all = torch.cat([x_all, vf_true_col], dim=1)

                x = x_all[:, x_idx]
                # Include pos at the beginning of x for scaling
                # x_with_pos = torch.cat([pos, x], dim=1)
                y = self._build_y_vector(g).view(1, -1)

                spos.partial_fit(pos.cpu().numpy())
                sx.partial_fit(x.cpu().numpy())
                sy.partial_fit(y.cpu().numpy())

                if self.edge_attr_mode != "none":
                    ea = _compute_edge_distance(pos, g.edge_index)
                    sedge.partial_fit(ea.cpu().numpy())

            with open(osp.join(self.processed_dir, "scalers.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "scaler": self.scaler,
                        "pos": spos,
                        "x": sx,
                        "y": sy,
                        "edge_attr": sedge,
                        "x_labels": self.x_labels,
                        "x_labels_all": self.x_labels_all,
                        "x_mode": self.x_mode,
                        "y_labels": self.y_labels,
                        "y_labels_all": self.y_labels_all,
                    },
                    f,
                )

        dataset: List[Data] = []

        for p in tqdm(graph_paths, desc="Build dataset"):
            d = torch.load(p)
            g = _safe_dict_to_data(d)

            pos = g.pos.to(torch.float32)
            material = g.material.to(torch.float32)
            euler = material[:, 0:3]  # always keep for optional edge transforms

            if self.x_mode == "euler":
                x_all = material
                quat = None
            elif self.x_mode == "quat":
                quat = euler_bunge_zxz_to_quat(euler, degrees=True).to(torch.float32)
                phase_size = material[:, 3:5]
                x_all = torch.cat([quat, phase_size], dim=1)
            else:
                raise ValueError(f"Unknown x_mode={self.x_mode}")

            # Append per-graph scalar (`graph_attr[4]`) as a constant extra node feature column.
            if not hasattr(g, "graph_attr"):
                raise ValueError("Graph is missing `graph_attr`; cannot append vf_true to node features.")
            if len(g.graph_attr) <= 4:
                raise ValueError(
                    f"`graph_attr` is too short (len={len(g.graph_attr)}); expected `graph_attr[4]` for vf_true."
                )
            vf_true = _as_float(g.graph_attr[4])
            vf_true_col = torch.full(
                (x_all.shape[0], 1),
                vf_true,
                dtype=torch.float32,
                device=x_all.device,
            )
            x_all = torch.cat([x_all, vf_true_col], dim=1)

            x = x_all[:, x_idx]
            # Include pos at the beginning of x for the dataset
            # x_with_pos = torch.cat([pos, x], dim=1)
            y = self._build_y_vector(g)

            out = Data(pos=pos, edge_index=g.edge_index, x=x, y=y, num_nodes=int(pos.shape[0]))
            # Keep graph-level attributes for stratified splitting / analysis.
            # Convention used by your dataset:
            # - graph_attr[0] = volume fraction (vf) in {0.1, 0.45, 0.9}
            # - graph_attr[2] = domain size in {1, 2, 4, 8} (as float)
            # - graph_attr[4] = true volume fraction (vf_true) appended to every node feature row
            if hasattr(g, "graph_attr"):
                out.graph_attr = g.graph_attr

            # edge_attr base
            if self.edge_attr_mode == "none":
                pass
            elif self.edge_attr_mode == "distance":
                out.edge_attr = _compute_edge_distance(out.pos, out.edge_index)
            else:
                raise ValueError(f"Unknown edge_attr_mode={self.edge_attr_mode}")

            # augment edge_attr
            if self.extra_edge_attrs:
                out.euler = euler
                # Make phase available for optional edge transforms (material[:,3] is phase).
                out.phase = material[:, 3]
                # If x is quaternion-based (or x_labels exclude quaternion columns), provide full quats for transforms.
                if quat is not None:
                    out.quat = quat
                if "euler_mismatch" in self.extra_edge_attrs:
                    out = EulerAngleMismatch(degrees=True, reduce="l2")(out)
                if "quat_angle" in self.extra_edge_attrs:
                    if not hasattr(out, "quat"):
                        out.quat = euler_bunge_zxz_to_quat(out.euler, degrees=True)
                    out = QuaternionAngleMismatch()(out)
                if "quat_rel" in self.extra_edge_attrs:
                    if not hasattr(out, "quat"):
                        out.quat = euler_bunge_zxz_to_quat(out.euler, degrees=True)
                    out = QuaternionRelativeRotation()(out)
                if "phase_pair" in self.extra_edge_attrs:
                    out = PhasePairCode()(out)

            # normalization
            if self.scaler != "none":
                out.pos = torch.from_numpy(spos.transform(out.pos.cpu().numpy()).astype(np.float32))
                out.x = torch.from_numpy(sx.transform(out.x.cpu().numpy()).astype(np.float32))
                out.y = torch.from_numpy(sy.transform(out.y.view(1, -1).cpu().numpy()).astype(np.float32)).view(-1)
                if hasattr(out, "edge_attr") and self.edge_attr_mode != "none":
                    out.edge_attr = torch.from_numpy(sedge.transform(out.edge_attr.cpu().numpy()).astype(np.float32))

            dataset.append(out)

        attrs = {
            "x_labels": self.x_labels,
            "x_labels_all": self.x_labels_all,
            "x_mode": self.x_mode,
            "y_labels": self.y_labels,
            "y_labels_all": self.y_labels_all,
            "y_labels_raw": self.y_labels_raw,
            "edge_attr_mode": self.edge_attr_mode,
            "extra_edge_attrs": self.extra_edge_attrs,
            "graphs_dir": self.graphs_dir,
            "graphs_glob": self.graphs_glob,
        }
        SimplePickleWriter(dataset, self.pickle_basedir, self.label, use_subdir=True, attrs=attrs)

    def _build_y_vector(self, g: Data) -> torch.Tensor:
        curves = {
            "stress_strain_x": self._extract_stress_curve(g, "stress_strain_x"),
            "stress_strain_y": self._extract_stress_curve(g, "stress_strain_y"),
        }
        needs_scalar = any(name not in curves for name in self.y_labels)
        scalar_vals: Optional[torch.Tensor] = None
        scalar_idx: Dict[str, int] = {}
        if needs_scalar:
            if not hasattr(g, "y"):
                raise ValueError("Graph is missing `y`; cannot add scalar targets.")
            y_all, y_labels_aug = self._augment_y_with_derived(g.y, self.y_labels_raw)
            scalar_vals = y_all
            scalar_idx = {k: i for i, k in enumerate(y_labels_aug)}
        parts: List[torch.Tensor] = []
        for name in self.y_labels:
            if name not in curves:
                if scalar_vals is None or name not in scalar_idx:
                    raise ValueError(
                        f"Unknown y label '{name}'. Available: {list(curves) + list(scalar_idx)}"
                    )
                parts.append(scalar_vals[scalar_idx[name]].view(1))
            else:
                parts.append(curves[name])
        return torch.cat(parts, dim=0)

    def _extract_stress_curve(self, g: Data, field: str) -> torch.Tensor:
        if not hasattr(g, field):
            raise ValueError(f"Graph is missing `{field}`; cannot build vector outputs.")
        curve = getattr(g, field)
        if not isinstance(curve, torch.Tensor):
            curve = torch.as_tensor(curve)
        curve = curve.to(torch.float32)
        if curve.ndim != 2 or curve.shape[0] != self.STRESS_STRAIN_LEN or curve.shape[1] < 2:
            raise ValueError(
                f"`{field}` must have shape ({self.STRESS_STRAIN_LEN}, >=2); got {tuple(curve.shape)}."
            )
        # Column 1 is stress; column 0 is strain (kept in data but not predicted here).
        # Select only the requested rows to reduce the vector length.
        idx = torch.as_tensor(self.STRESS_STRAIN_ROWS_0B, dtype=torch.long, device=curve.device)
        return curve[idx, 1].view(-1)


def _as_float(x) -> float:
    # Robustly convert tensors/arrays/python scalars to float
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().view(-1)[0].item())
    return float(x)


def _canonical_vf(vf: float) -> float:
    # Avoid float quirks when keys come from tensors
    return float(np.round(vf, 6))


def _canonical_size(size: float) -> int:
    # Dataset uses size encoded as float in {1,2,4,8}; map to int with tolerance.
    s = _as_float(size)
    for cand in (1.0, 2.0, 4.0, 8.0):
        if np.isclose(s, cand, rtol=0.0, atol=1e-6):
            return int(cand)
    # Fallback: nearest int (still deterministic)
    return int(np.round(s))


def _canonical_sve_id(sve_id) -> int:
    # Robustly convert tensors/arrays/python scalars to int
    if isinstance(sve_id, torch.Tensor):
        return int(sve_id.detach().cpu().view(-1)[0].item())
    return int(sve_id)


def _get_vf_sve_size_block(g: Data) -> tuple[float, int, int, int]:
    """
    Return (vf, sve_id, size, block_idx) from graph_attr and optional `cell_block_idx`.

    Conventions in this dataset:
    - g.graph_attr[0] = volume fraction (vf)
    - g.graph_attr[1] = SVE/image id (all partitions across sizes share this)
    - g.graph_attr[2] = partition grid size in {1,2,4,8}
    - g.cell_block_idx = sub-partition id within that grid, expected 0..(size^2-1)

    Ordering convention (as provided):
    block 0 is bottom-left, increases to the right, then upward.
    Example for 2x2:
        2, 3
        0, 1
    """
    if not hasattr(g, "graph_attr"):
        raise ValueError(
            "Graph is missing `graph_attr`, so we cannot split/group by vf/sve/size. "
            "Ensure graphs contain graph_attr and CP2D.process copies it onto Data."
        )
    ga = g.graph_attr
    vf = _canonical_vf(_as_float(ga[0]))
    sve_id = _canonical_sve_id(ga[1])
    size = _canonical_size(_as_float(ga[2]))
    block_idx = int(getattr(g, "cell_block_idx", 0))
    return vf, sve_id, size, block_idx


def _filter_by_vfs_images(dataset: List[Data], allowed_vfs: Optional[Sequence[float]]) -> List[Data]:
    if not allowed_vfs:
        return dataset
    allowed = [_canonical_vf(float(v)) for v in allowed_vfs]
    out: List[Data] = []
    missing_graph_attr = 0
    for g in dataset:
        if not hasattr(g, "graph_attr"):
            missing_graph_attr += 1
            continue
        vf, _, _, _ = _get_vf_sve_size_block(g)
        if any(np.isclose(vf, a, rtol=0.0, atol=1e-6) for a in allowed):
            out.append(g)
    if missing_graph_attr:
        raise ValueError(
            f"{missing_graph_attr} graphs were missing graph_attr; cannot filter/split reliably."
        )
    return out


def split_by_sve_images_per_vf(
    dataset,
    *,
    n_train_images: int = 26,
    n_val_images: int = 2,
    n_test_images: int = 2,
    seed: int = 0,
    allowed_vfs: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
) -> tuple[List[Data], List[Data], List[Data]]:
    """
    Split dataset by **SVE/image id** within each volume fraction (vf).

    For each vf:
    - shuffle unique SVE ids (seeded)
    - take first n_train_images for train, next n_val_images for val, next n_test_images for test
    - include **all graph partitions** (all sizes and sub-partitions) for each chosen SVE id

    This guarantees that all components of an image stay together across splits.
    Within each (vf, sve_id) image group, graphs are ordered deterministically by:
      (size ascending, block_idx ascending).
    """
    if n_train_images < 0 or n_val_images < 0 or n_test_images < 0:
        raise ValueError("n_train_images/n_val_images/n_test_images must be non-negative.")

    all_data: List[Data] = list(dataset)
    all_data = _filter_by_vfs_images(all_data, allowed_vfs)
    if not all_data:
        raise ValueError("Dataset is empty after filtering.")

    # Group by (vf, sve_id)
    by_vf_sve: Dict[tuple[float, int], List[Data]] = {}
    vfs_seen: set[float] = set()
    for g in all_data:
        vf, sve_id, _, _ = _get_vf_sve_size_block(g)
        vfs_seen.add(vf)
        by_vf_sve.setdefault((vf, sve_id), []).append(g)

    rng = np.random.default_rng(int(seed))

    # Determine split by SVE ids within each vf
    image_order_per_vf: Dict[float, List[int]] = {}
    splits_sve: Dict[str, Dict[float, List[int]]] = {"train": {}, "val": {}, "test": {}}

    train: List[Data] = []
    val: List[Data] = []
    test: List[Data] = []

    for vf in sorted(vfs_seen):
        sve_ids = sorted({sve_id for (vv, sve_id) in by_vf_sve.keys() if vv == vf})
        if len(sve_ids) < (n_train_images + n_val_images + n_test_images):
            raise ValueError(
                f"vf={vf} has only {len(sve_ids)} unique SVE ids, but split requires "
                f"{n_train_images + n_val_images + n_test_images}."
            )

        sve_ids_shuffled = list(sve_ids)
        rng.shuffle(sve_ids_shuffled)
        image_order_per_vf[vf] = list(sve_ids_shuffled)

        train_ids = sve_ids_shuffled[:n_train_images]
        val_ids = sve_ids_shuffled[n_train_images : n_train_images + n_val_images]
        test_ids = sve_ids_shuffled[
            n_train_images + n_val_images : n_train_images + n_val_images + n_test_images
        ]

        splits_sve["train"][vf] = list(train_ids)
        splits_sve["val"][vf] = list(val_ids)
        splits_sve["test"][vf] = list(test_ids)

        def _ordered_graphs(vf_: float, sve_id_: int) -> List[Data]:
            graphs = list(by_vf_sve[(vf_, sve_id_)])
            graphs.sort(key=lambda gg: (_get_vf_sve_size_block(gg)[2], _get_vf_sve_size_block(gg)[3]))
            return graphs

        for sve_id in train_ids:
            train.extend(_ordered_graphs(vf, sve_id))
        for sve_id in val_ids:
            val.extend(_ordered_graphs(vf, sve_id))
        for sve_id in test_ids:
            test.extend(_ordered_graphs(vf, sve_id))

    if save_path:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        payload = {
            "seed": int(seed),
            "n_train_images": int(n_train_images),
            "n_val_images": int(n_val_images),
            "n_test_images": int(n_test_images),
            "allowed_vfs": list(allowed_vfs) if allowed_vfs is not None else None,
            "image_order_per_vf": {str(k): v for k, v in image_order_per_vf.items()},
            "splits_sve_ids": {
                split: {str(vf): ids for vf, ids in d.items()} for split, d in splits_sve.items()
            },
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)

    return train, val, test


def split_by_sve_json(
    dataset,
    split_json_path: str,
    *,
    allowed_vfs: Optional[Sequence[float]] = None,
) -> tuple[List[Data], List[Data], List[Data]]:
    """
    Recreate train/val/test subsets from a saved split JSON payload written by
    `split_by_sve_images_per_vf`.
    """
    with open(split_json_path, "r") as f:
        payload = json.load(f)

    if "splits_sve_ids" not in payload:
        raise ValueError(f"{split_json_path} does not contain `splits_sve_ids`.")

    all_data: List[Data] = list(dataset)
    all_data = _filter_by_vfs_images(all_data, allowed_vfs)
    if not all_data:
        raise ValueError("Dataset is empty after filtering.")

    by_vf_sve: Dict[tuple[float, int], List[Data]] = {}
    for g in all_data:
        vf, sve_id, _, _ = _get_vf_sve_size_block(g)
        by_vf_sve.setdefault((vf, sve_id), []).append(g)

    split_defs = payload["splits_sve_ids"]
    subsets: Dict[str, List[Data]] = {"train": [], "val": [], "test": []}

    def _ordered_graphs(vf_: float, sve_id_: int) -> List[Data]:
        key = (vf_, sve_id_)
        if key not in by_vf_sve:
            raise ValueError(
                f"Split file references vf={vf_}, sve_id={sve_id_}, but no matching graphs were found."
            )
        graphs = list(by_vf_sve[key])
        graphs.sort(key=lambda gg: (_get_vf_sve_size_block(gg)[2], _get_vf_sve_size_block(gg)[3]))
        return graphs

    for split_name in ("train", "val", "test"):
        vf_to_ids = split_defs.get(split_name, {})
        for vf_key, sve_ids in vf_to_ids.items():
            vf = _canonical_vf(float(vf_key))
            if allowed_vfs and not any(np.isclose(vf, _canonical_vf(float(a)), rtol=0.0, atol=1e-6) for a in allowed_vfs):
                continue
            for sve_id in sve_ids:
                subsets[split_name].extend(_ordered_graphs(vf, int(sve_id)))

    return subsets["train"], subsets["val"], subsets["test"]


def _print_image_split_stats(name: str, subset: List[Data]) -> None:
    """
    Print split stats by vf, image (sve_id), and size.
    """
    # counts[vf][sve_id][size] = n_graphs
    counts: Dict[float, Dict[int, Dict[int, int]]] = {}
    for g in subset:
        vf, sve_id, size, _ = _get_vf_sve_size_block(g)
        counts.setdefault(vf, {}).setdefault(sve_id, {})[size] = (
            counts.setdefault(vf, {}).setdefault(sve_id, {}).get(size, 0) + 1
        )

    total = len(subset)
    print(f"{name}: n_graphs={total}")
    for vf in sorted(counts.keys()):
        sve_ids = sorted(counts[vf].keys())
        print(f"  vf={vf}: n_images={len(sve_ids)}")
        # aggregate by size across images
        size_totals: Dict[int, int] = {}
        for sve_id in sve_ids:
            for size, c in counts[vf][sve_id].items():
                size_totals[size] = size_totals.get(size, 0) + c
        for size, c in sorted(size_totals.items(), key=lambda kv: kv[0]):
            print(f"    size={size}: {c}")


def _get_vf_size(g: Data) -> tuple[float, int]:
    if not hasattr(g, "graph_attr"):
        raise ValueError(
            "Graph is missing `graph_attr`, so we cannot stratify by vf/size. "
            "Ensure graphs contain graph_attr and CP2D.process copies it onto Data."
        )
    ga = g.graph_attr
    # graph_attr is a vector; vf at [0], size at [2]
    vf = _canonical_vf(_as_float(ga[0]))
    size = _canonical_size(_as_float(ga[2]))
    return vf, size


def _filter_by_vfs(dataset: List[Data], allowed_vfs: Optional[Sequence[float]]) -> List[Data]:
    if not allowed_vfs:
        return dataset
    allowed = [_canonical_vf(float(v)) for v in allowed_vfs]
    out: List[Data] = []
    missing_graph_attr = 0
    for g in dataset:
        if not hasattr(g, "graph_attr"):
            missing_graph_attr += 1
            continue
        vf, _ = _get_vf_size(g)
        if any(np.isclose(vf, a, rtol=0.0, atol=1e-6) for a in allowed):
            out.append(g)
    if missing_graph_attr:
        raise ValueError(
            f"{missing_graph_attr} graphs were missing graph_attr; cannot filter/split reliably."
        )
    return out


def stratified_split_by_size_vf(
    dataset,
    *,
    perc_train: float = 0.9,
    perc_val: float = 0.05,
    perc_test: float = 0.05,
    seed: int = 0,
    allowed_vfs: Optional[Sequence[float]] = None,
) -> tuple[List[Data], List[Data], List[Data]]:
    """
    Deterministic stratified split where each (size, vf) group keeps ~perc_train/val/test proportions.

    This is designed for CP2D where:
    - g.graph_attr[2] ∈ {1,2,4,8} encodes domain size
    - g.graph_attr[0] ∈ {0.1,0.45,0.9} encodes volume fraction (vf)
    """
    if perc_train <= 0 or perc_val < 0 or perc_test < 0:
        raise ValueError("Invalid split fractions.")
    if not np.isclose(perc_train + perc_val + perc_test, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("perc_train + perc_val + perc_test must equal 1.0")

    # Materialize (SimplePickleDataset is iterable)
    all_data: List[Data] = list(dataset)
    all_data = _filter_by_vfs(all_data, allowed_vfs)
    if not all_data:
        raise ValueError("Dataset is empty after filtering.")

    groups: Dict[tuple[int, float], List[int]] = {}
    for i, g in enumerate(all_data):
        vf, size = _get_vf_size(g)
        key = (size, vf)
        groups.setdefault(key, []).append(i)

    rng = np.random.default_rng(int(seed))

    train: List[Data] = []
    val: List[Data] = []
    test: List[Data] = []

    for key, idxs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)

        # Make train the closest integer to perc_train*n; split the remainder as evenly as possible.
        n_train = int(np.round(n * perc_train))
        n_train = max(0, min(n, n_train))
        rem = n - n_train
        n_val = int(np.floor(rem * (perc_val / (perc_val + perc_test))) if (perc_val + perc_test) > 0 else 0)
        n_val = max(0, min(rem, n_val))
        n_test = rem - n_val

        train.extend(all_data[i] for i in idxs[:n_train])
        val.extend(all_data[i] for i in idxs[n_train : n_train + n_val])
        test.extend(all_data[i] for i in idxs[n_train + n_val : n_train + n_val + n_test])

    return train, val, test


def _print_split_stats(name: str, subset: List[Data]) -> None:
    counts: Dict[tuple[int, float], int] = {}
    for g in subset:
        vf, size = _get_vf_size(g)
        counts[(size, vf)] = counts.get((size, vf), 0) + 1
    total = len(subset)
    print(f"{name}: n={total}")
    for (size, vf), c in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  size={size} vf={vf}: {c} ({(c / total * 100.0):.2f}%)")
