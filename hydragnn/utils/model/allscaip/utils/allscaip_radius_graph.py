"""
Bidirectional kNN radius graph builder for AllScAIP.

Modified from MinDScAIP (credit: Ryan Liu) and trimmed for HydraGNN: removed
chunked-construction, low-memory soft-rank, padding, torch.compile fast-paths,
and per-system batched padding helpers — none of which are exercised by the
HydraGNN integration (we always run unpadded, eager forward).

The output is the dense, padded ``(N, k_max, ...)`` tensor layout that
AllScAIP's transformer attention blocks consume; this is fundamentally
different from a sparse PyG ``edge_index`` and is a load-bearing part of the
model architecture, not just plumbing.
"""

from __future__ import annotations

from typing import Any

import torch

AtomicData = Any

from hydragnn.utils.model.escaip.utils.radius_graph import (
    hard_rank,
    safe_norm,
    soft_rank,
)


def build_radius_graph(
    pos: torch.Tensor,
    cell: torch.Tensor,
    image_id: torch.Tensor,
    cutoff: float,
    start_index: int,
    device: torch.device,
    k: int,
    soft: bool,
    sigmoid_scale: float,
    lse_scale: float,
    compute_dist_pairwise: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Construct the biknn radius graph for one system.

    Returns
    -------
    index1, index2 : (E,) source / destination atom indices (with ``start_index`` offset)
    index1_rank, index2_rank : (E,) within-source / within-destination neighbor rank by envelope
    disp : (E, 3) displacement vectors
    env : (E,) envelope values in [0, 1)
    dist_pairwise : (N, N) min-image pairwise distances, or ``None``
    """
    N = pos.size(0)
    M = image_id.size(0)
    # PBC-aware displacements: src_pos[i, m] = pos[i] + image_id[m] @ cell
    src_pos = pos[:, None] + torch.mm(image_id, cell)[None, :]
    disp = src_pos[None, :, :, :] - pos[:, None, None, :]
    dist = safe_norm(disp, dim=-1)
    dist_T = dist.transpose(0, 1).contiguous()

    dist_pairwise = dist.min(dim=2)[0] if compute_dist_pairwise else None

    if soft:
        src_ranks = soft_rank(dist.view(N, N * M), sigmoid_scale).view(N, N, M)
        dst_ranks = (
            soft_rank(dist_T.view(N, N * M), sigmoid_scale)
            .view(N, N, M)
            .transpose(0, 1)
        )
        env = torch.stack([src_ranks / k, dst_ranks / k, dist / cutoff], dim=0)
        env = lse_scale * torch.logsumexp(env / lse_scale, dim=0)
    else:
        src_ranks = hard_rank(dist.view(N, N * M)).view(N, N, M)
        dst_ranks = hard_rank(dist_T.view(N, N * M)).view(N, N, M).transpose(0, 1)
        env = torch.stack([src_ranks / k, dst_ranks / k, dist / cutoff], dim=0)
        env = torch.amax(env, dim=0)

    # zero out self-loop envelopes
    env.masked_fill_(dist == 0.0, 0.0)

    # rank each (src, dst, image) edge within source-side and destination-side
    # neighbor lists by envelope value
    index = torch.arange(N, device=device)[:, None]
    ranks = torch.arange(M * N, device=device)[None, :]
    index1_rank = torch.full((N, N, M), -1, device=device, dtype=torch.long)
    src_argsort = torch.argsort(env.view(N, N * M), dim=1)
    index1_rank[index, src_argsort // M, src_argsort % M] = ranks
    index2_rank = torch.full((N, N, M), -1, device=device, dtype=torch.long)
    dst_argsort = torch.argsort(env.transpose(0, 1).reshape(N, N * M), dim=1)
    index2_rank[dst_argsort // M, index, dst_argsort % M] = ranks

    # filter to envelope < 1 (= within cutoff and within k nearest in both directions)
    mask = env < 1.0
    index1, index2, index3 = torch.where(mask)
    return (
        index1 + start_index,
        index2 + start_index,
        index1_rank[index1, index2, index3],
        index2_rank[index1, index2, index3],
        disp[index1, index2, index3],
        env[index1, index2, index3],
        dist_pairwise,
    )


def batched_radius_graph(
    pos_list: list[torch.Tensor],
    cell_list: list[torch.Tensor],
    image_id_list: list[torch.Tensor],
    num_atoms: int,
    slices: list[int],
    knn_k: int,
    knn_soft: bool,
    knn_sigmoid_scale: float,
    knn_lse_scale: float,
    cutoff: float,
    device: torch.device,
    compute_dist_pairwise: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Batch the per-system biknn graphs and assemble dense ``(N, k_max, ...)`` tensors."""
    results = [
        build_radius_graph(
            pos,
            cell,
            image_id,
            cutoff,
            start_idx,
            device,
            knn_k,
            knn_soft,
            knn_sigmoid_scale,
            knn_lse_scale,
            compute_dist_pairwise,
        )
        for pos, cell, image_id, start_idx in zip(
            pos_list, cell_list, image_id_list, slices
        )
    ]
    (
        index1_list,
        index2_list,
        index1_rank_list,
        index2_rank_list,
        disp_list,
        env_list,
        dist_blocks,
    ) = zip(*results)

    index1 = torch.cat(index1_list)
    index2 = torch.cat(index2_list)
    index1_rank = torch.cat(index1_rank_list)
    index2_rank = torch.cat(index2_rank_list)
    disp = torch.cat(disp_list)
    env = torch.cat(env_list)

    dist_pairwise = torch.block_diag(*dist_blocks) if compute_dist_pairwise else None

    # auto-size the per-node neighbor padding to the actual maximum
    knn_pad_size = int(max(index1_rank.max().item(), index2_rank.max().item())) + 1

    # initialize the padded tensors
    padded_index = (
        torch.arange(num_atoms, device=device)
        .view(-1, 1)
        .expand(num_atoms, knn_pad_size)
    )
    padded_rank = (
        torch.arange(knn_pad_size, device=device)
        .view(1, -1)
        .expand(num_atoms, knn_pad_size)
    )
    padded_disp = torch.zeros((num_atoms, knn_pad_size, 3), device=device)
    src_env = torch.full((num_atoms, knn_pad_size), torch.inf, device=device)
    dst_env = torch.full((num_atoms, knn_pad_size), torch.inf, device=device)
    edge_index = torch.stack([padded_index, padded_index], dim=0)
    src_index = torch.stack([padded_index, padded_rank], dim=0)
    dst_index = torch.stack([padded_index, padded_rank], dim=0)

    # scatter the real edges into the padded layout
    padded_disp[index1, index1_rank] = disp
    src_env[index1, index1_rank] = env
    dst_env[index2, index2_rank] = env
    edge_index[0, index1, index1_rank] = index1
    edge_index[1, index1, index1_rank] = index2
    # cross-references between source-neighbor and destination-neighbor layouts
    src_index[0, index1, index1_rank] = index2
    src_index[1, index1, index1_rank] = index2_rank
    dst_index[0, index2, index2_rank] = index1
    dst_index[1, index2, index2_rank] = index1_rank

    return (
        dist_pairwise,
        padded_disp,
        src_env,
        dst_env,
        src_index,
        dst_index,
        edge_index,
    )


def biknn_radius_graph(
    data: AtomicData,
    cutoff: float,
    knn_k: int,
    knn_soft: bool,
    knn_sigmoid_scale: float,
    knn_lse_scale: float,
    device: torch.device,
    compute_dist_pairwise: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build the biknn radius graph for a (possibly batched) system. Computes the
    PBC image identifier list (skipping it entirely when no system in the batch
    is periodic) and dispatches to :func:`batched_radius_graph`.
    """
    slices, _, _, natoms_list = data.get_batch_stats()
    if slices is None or natoms_list is None:
        # Unbatched single-system input (e.g., from ASE calculator): synthesize stats.
        n = data.pos.shape[0]
        natoms_list = [n]
        slices = {"pos": torch.tensor([0, n], device=data.pos.device)}

    pos_list: list[torch.Tensor] = list(torch.split(data.pos, natoms_list, dim=0))
    num_graphs = len(natoms_list)

    pbc_any = data.pbc.any().item()
    if pbc_any:
        # Replicate cells along each PBC axis enough to cover ``cutoff``
        cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
        cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

        rep_a1 = torch.ceil(cutoff * safe_norm(cross_a2a3 / cell_vol, dim=-1))
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        rep_a2 = torch.ceil(cutoff * safe_norm(cross_a3a1 / cell_vol, dim=-1))
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        rep_a3 = torch.ceil(cutoff * safe_norm(cross_a1a2 / cell_vol, dim=-1))

        rep_a1 = rep_a1.masked_fill(data.pbc[:, 0] == 0, 0).tolist()
        rep_a2 = rep_a2.masked_fill(data.pbc[:, 1] == 0, 0).tolist()
        rep_a3 = rep_a3.masked_fill(data.pbc[:, 2] == 0, 0).tolist()

        image_id_list: list[torch.Tensor] = [
            torch.cartesian_prod(
                *[
                    torch.arange(
                        -rep,
                        rep + 1,
                        device=device,
                        dtype=torch.get_default_dtype(),
                    )
                    for rep in reps
                ]
            )
            for reps in zip(rep_a1, rep_a2, rep_a3)
        ]
    else:
        identity_image = torch.zeros(
            (1, 3), device=device, dtype=torch.get_default_dtype()
        )
        image_id_list = [identity_image for _ in range(num_graphs)]

    cell_list: list[torch.Tensor] = list(data.cell)

    return batched_radius_graph(
        pos_list,
        cell_list,
        image_id_list,
        data.num_nodes,
        slices["pos"],
        knn_k,
        knn_soft,
        knn_sigmoid_scale,
        knn_lse_scale,
        cutoff,
        device,
        compute_dist_pairwise,
    )
