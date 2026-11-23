"""manifest.py — single source of truth for all cells in the v4 ISAAC study.

The v4 study targets the failure modes diagnosed from v3:

  - v3 leaderboard: pn_R²≈0.62, pn_spear≈0.79, but pgmax_R²≈−3.6, top10_slope≈0.15.
  - Root cause: per-grain MSE shrinks predictions (std_pred = 0.79·std_true),
    and per-graph max = max-of-shrunken-grains amplifies the bias.
  - v3 explicitly turned OFF the per-graph head (`use_per_graph_target=False`)
    and graph-level conditioning (`graph_attr_source=None`), so the model had
    neither an architectural lever to predict per-graph max directly nor any
    information about SVE size. Fixing those two knobs is the v4 thesis.

v4 has four sequential tracks:

  Track A  — Architecture & graph-level conditioning   7 cells
  Track G  — Aux Gumbel-distribution heads (gates on A) 4 cells
  Track L  — Loss / tail emphasis (gates on A+G)        4 cells
  Track P  — Promote winners to extra seeds            up to 8 cells × 2 extra
  Track E  — Final 5-split × 5-init validation         1 config × 25 reps

A new post-processing step (`calibration.py`) applies an isotonic-style
quantile mapping fit on val, applied at test, that flips pgmax_R² from
−3.6 to ~+0.15 *on the v3 numbers* — so calibration is run on every v4 cell
at analysis time and reported alongside raw scores.

Total wall on 2× V100 in parallel ≈ 6–10 hours.

Each cell is a dict consumable by `_runner_utils.run_cell()`. BASE_RECIPE
is the v3-final tuned Path-F recipe (lr=1e-4, ROP, MSE, EarlyStopping
patience=200, Checkpoint=True, num_epoch=2000).
"""
from __future__ import annotations
from fip_release_paths import release_root
from typing import Any, Dict, List

BASE_RECIPE: Dict[str, Any] = {
    "Verbosity": {"level": 1},
    "NeuralNetwork": {
        "Architecture": {
            "mpnn_type": "SAGE",
            "hidden_dim": 16,
            "num_conv_layers": 6,
            "activation_function": "elu",
            "pe_dim": 2,
            "max_neighbours": 5,
            "radius": 7,
            "graph_pooling": "max",
            "global_attn_engine": "",
            "global_attn_type": "",
            "edge_features": [],
        },
        "Training": {
            "num_epoch": 800,
            "loss_function_type": "mse",
            "batch_size": 32,
            "EarlyStopping": True,
            "patience": 100,
            "Checkpoint": True,
            "checkpoint_warmup": 5,
            "Optimizer": {"type": "AdamW", "learning_rate": 0.0001},
            "weight_decay": 0.0,
            "lr_scheduler": "reduce_on_plateau",
            "lr_scheduler_factor": 0.3,
            "lr_scheduler_patience": 15,
            "lr_scheduler_min_lr": 1e-07,
        },
    },
    "Recipe": {
        "size_bucket": None,
        "stratify_by_bucket": True,
        "split_fracs": [0.9, 0.05, 0.05],
        "use_per_grain_target": True,
        "use_per_graph_target": True,
        "per_grain_target_field": "fs_fip",
        "per_grain_target_col": 0,
        "per_graph_target_aggr": "max",
        "node_task_weight": 1.0,
        "per_grain_transform": "zscore",
        "per_graph_transform": "log1p",
        "graph_attr_source": None,
        "graph_attr_conditioning_mode": "concat_node",
        "use_x_blocks": [
            "quat",
            "schmid12",
            "volume",
            "surface_area",
            "perimeter",
            "degree",
            "Iuu",
            "Ivv",
            "Iuv",
            "J_polar",
            "I1",
            "I2",
            "voxel_geom",
        ],
        "use_edge_cols": ["edge_length"],
        "drop_periodic_edges": False,
        "loss_recipe": None,
        "aux_targets": None,
        "aux_task_weights": None,
    },
}
TRACK_A: List[Dict[str, Any]] = [
    {
        "tag": "A0_control",
        "extra": {"Recipe": {"use_per_graph_target": False, "graph_attr_source": None}},
        "note": "Replicates v3 F1 (per-grain only, no graph_attr). Sanity baseline.",
    },
    {
        "tag": "A1_pg_head",
        "extra": {"Recipe": {"use_per_graph_target": True, "graph_attr_source": None}},
        "note": "Adds per-graph MAX head. Direct regression target avoids order-statistic compression.",
    },
    {
        "tag": "A2_logN",
        "extra": {
            "Recipe": {
                "use_per_graph_target": False,
                "graph_attr_source": "log_n_nodes",
            }
        },
        "note": "log(N_grains) as graph attribute; absorbs LS scaling without graph head.",
    },
    {
        "tag": "A3_pg_logN",
        "extra": {
            "Recipe": {"use_per_graph_target": True, "graph_attr_source": "log_n_nodes"}
        },
        "note": "A1+A2 combined. Most likely v4 winner if A1 alone shows lift.",
    },
    {
        "tag": "A4_pg_mech",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "mech_compact",
            }
        },
        "note": "20-d CPFEM mech responses (monotonic_plastic + cyclic_plastic + cyclic_elastic). Memory says +5x parity on cp3d_v2. INTERNAL ONLY — plastic components risk leakage; not paper-safe.",
    },
    {
        "tag": "A4e_pg_mechE",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "mech_compact_elastic",
            }
        },
        "note": "11-d COMPACT ELASTIC-ONLY mech response (only dims with meaningful inter-SVE variance — see dataset.py:'mech_compact_elastic'). Publication-safe — no plastic FIP coupling. Compare against A4 to quantify how much of mech_compact's lift is plastic leakage vs. real.",
    },
    {
        "tag": "A5_pg_bucket",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
            }
        },
        "note": "5-d one-hot of SVE size bucket. Limited to in-distribution sizes.",
    },
    {
        "tag": "A6_pg_aggmech",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "agg_plus_mech",
            }
        },
        "note": "38-d aggregate microstructure + mech responses. Most informative graph_attr available.",
    },
]
TRACK_G: List[Dict[str, Any]] = [
    {
        "tag": "G0_aux_mu",
        "extra": {"Recipe": {"aux_targets": ["gumbel_mu"], "aux_task_weights": [0.5]}},
        "note": "Aux head: per-grain Gumbel μ alongside MAX. Cheapest aux.",
    },
    {
        "tag": "G1_aux_mu_beta",
        "extra": {
            "Recipe": {
                "aux_targets": ["gumbel_mu", "gumbel_beta"],
                "aux_task_weights": [0.5, 0.5],
            }
        },
        "note": "μ + β: enables Gumbel reconstruction of grain-MAX from (μ, β).",
    },
    {
        "tag": "G2_aux_p99_top",
        "extra": {
            "Recipe": {
                "aux_targets": ["p99", "topk_mean"],
                "aux_task_weights": [0.3, 0.3],
            }
        },
        "note": "Smoother tail-statistic targets (p99, top-k mean). Easier to fit than max.",
    },
    {
        "tag": "G3_aux_full",
        "extra": {
            "Recipe": {
                "aux_targets": [
                    "std",
                    "p90",
                    "p99",
                    "topk_mean",
                    "gumbel_mu",
                    "gumbel_beta",
                ],
                "aux_task_weights": [0.3, 0.3, 0.3, 0.3, 0.5, 0.5],
            }
        },
        "note": "All aux columns. Maximal regularization signal at cost of training stability.",
    },
]
TRACK_L: List[Dict[str, Any]] = [
    {
        "tag": "L0_mse",
        "extra": {
            "NeuralNetwork": {"Training": {"loss_function_type": "mse"}},
            "Recipe": {"loss_recipe": None},
        },
        "note": "Control — same MSE as Track A.",
    },
    {
        "tag": "L1_huber",
        "extra": {
            "NeuralNetwork": {"Training": {"loss_function_type": "huber"}},
            "Recipe": {"loss_recipe": None},
        },
        "note": "HydraGNN-native huber. Bug-fixed: not via loss_recipe.",
    },
    {
        "tag": "L2_phtopk",
        "extra": {
            "NeuralNetwork": {"Training": {"loss_function_type": "mse"}},
            "Recipe": {"loss_recipe": "peak_huber_topk_05_15"},
        },
        "note": "Tail-aware peak-huber-topk. Routed via loss_recipe override.",
    },
    {
        "tag": "L3_phtopk_aggressive",
        "extra": {
            "NeuralNetwork": {"Training": {"loss_function_type": "mse"}},
            "Recipe": {"loss_recipe": "peak_huber_topk_05_30"},
        },
        "note": "Same family, more aggressive top-k weighting.",
    },
]
TRACK_B: List[Dict[str, Any]] = [
    {
        "tag": "B0_baseline",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "log_n_nodes",
                "loss_recipe": "mse_stdmatch_50",
            }
        },
        "note": "Baseline = A3*. Sanity check that running through TRACK_B harness reproduces existing A3_pg_logN_stdmatch_l50 metrics.",
    },
    {
        "tag": "B1_peak_topk_05_15",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "log_n_nodes",
                "loss_recipe": "peak_huber_topk_05_15",
            }
        },
        "note": "anik production winner: alpha=0.5, beta=0.35, gamma=0.15. Anik top-1 = 0.35 vs baseline 0.06; biggest theoretical lift.",
    },
    {
        "tag": "B2_peak_topk_05_25",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "log_n_nodes",
                "loss_recipe": "peak_huber_topk_05_25",
            }
        },
        "note": "More top-K emphasis: gamma=0.25; untested on fip_unified.",
    },
    {
        "tag": "B3_rank_huber_05",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "log_n_nodes",
                "loss_recipe": "rank_huber_05",
            }
        },
        "note": "Pure pairwise rank blend (alpha=0.5). Anik top-1 = 0.21.",
    },
    {
        "tag": "B4_listmle_05",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "log_n_nodes",
                "loss_recipe": "listmle_05",
            }
        },
        "note": "Plackett-Luce listwise NLL of true ranking. Smoke-tested only on anik (negative R^2 there, but anik dataset was corrupted at the time). Untested on clean fip_unified.",
    },
    {
        "tag": "B5_a5_stdmatch",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_50",
            }
        },
        "note": "A5 base (5-d size_bucket onehot) + same std-match. A5 had the highest top-1 (0.14) in the architecture ablation but lacked the std-match term; this combines both.",
    },
    {
        "tag": "C1_a5_stdmatch_l75",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
            }
        },
        "note": "A5 + larger stdmatch lambda=0.75. More within-SVE spread pressure.",
    },
    {
        "tag": "C2_a5_stdmatch_l100",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_100",
            }
        },
        "note": "A5 + extreme stdmatch lambda=1.00. Maximum within-SVE spread emphasis.",
    },
    {
        "tag": "C3_a5_peak_topk_05_15",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "peak_huber_topk_05_15",
            }
        },
        "note": "A5 + anik production loss (alpha=0.5, gamma=0.15). On A3 base it hurt top-1; with A5's bucket conditioning it might now help (the architecture-loss interaction matters).",
    },
    {
        "tag": "C4_a5_peak_topk_05_30",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "peak_huber_topk_05_30",
            }
        },
        "note": "A5 + peak_huber_topk with gamma=0.30 (more top-decile pressure).",
    },
    {
        "tag": "C5_a3_stdmatch_l100",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "log_n_nodes",
                "loss_recipe": "mse_stdmatch_100",
            }
        },
        "note": "A3 + lambda=1.00 control. If large lambda lifts A3's top-1 (0.06 -> 0.12 with l=0.5) further, lambda is the knob; if not, the bucket conditioning is doing the work.",
    },
    {
        "tag": "D1_a5_gumbel_50_50",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_gumbel_50_50",
            }
        },
        "note": "A5 + Gumbel moment match (lam_b=0.5, lam_m=0.5). Generalises stdmatch by adding mu-matching in the Gumbel-physics frame.",
    },
    {
        "tag": "D2_a5_gumbel_100_0",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_gumbel_100_0",
            }
        },
        "note": "A5 + Gumbel beta-only match (lam_b=1.0, lam_m=0). Equivalent to stdmatch with sqrt(6)/pi ~ 0.78 scaling and lambda=1; tests whether the Gumbel scaling factor matters at all vs stdmatch_50.",
    },
    {
        "tag": "D3_a5_gumbel_tw_05_30",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_gumbel_tw_05_30",
            }
        },
        "note": "A5 + Gumbel-CDF tail-weighted MSE (alpha=0.5, k=3.0). Smooth upper-tail emphasis, Gumbel-aware. Direct top-1 lever.",
    },
    {
        "tag": "D4_a5_gumbel_tw_08_30",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_gumbel_tw_08_30",
            }
        },
        "note": "A5 + aggressive Gumbel-CDF tail weight (alpha=0.8, k=3.0).",
    },
    {
        "tag": "D5_a5_stdmatch_pgGumbel",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_50",
                "per_graph_transform": "gumbel_uniform",
            }
        },
        "note": "A5 + B5's stdmatch + per-graph head trained in Gumbel-CDF uniform space (per_graph_transform=gumbel_uniform). The per-grain head keeps z-score normalisation; only the per-graph head sees Gumbel-shaped targets.",
    },
    {
        "tag": "E1_a5_stdmm_50_50",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmm_50_50",
            }
        },
        "note": "A5 + stdmatch (lam_std=0.5) + within-SVE max-margin (lam_mar=0.5, margin=0.5). Directly penalises argmax misses on the per-grain head.",
    },
    {
        "tag": "E2_a5_stdmm_50_100",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmm_50_100",
            }
        },
        "note": "A5 + stdmatch + aggressive max-margin (lam_mar=1.0).",
    },
    {
        "tag": "E3_a5_stdmm_75_50",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmm_75_50",
            }
        },
        "note": "A5 + C1's lam_std=0.75 + max-margin (lam_mar=0.5). Combines C1's beta-recovery improvement with direct argmax pressure.",
    },
    {
        "tag": "F1_a5_stdmatch_l75_globalz",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_grain_transform": "zscore",
            }
        },
        "note": "Same as C1 but per_grain_transform=zscore (global) rather than zscore_per_bucket. Tests claim from 11_grain_vs_svemax_qa.md that the per-grain MAX PDFs are size-invariant, hence per-bucket and global z-score are equivalent.",
    },
    {
        "tag": "G0_a5_C1_normedges",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "use_edge_cols": ["edge_length"],
            }
        },
        "note": "C1 with edge_attr now standardized via normalize_edges (default True). SAGE does not consume edge features, so this should match C1's metrics to sampling noise. Sanity check.",
    },
    {
        "tag": "G1_a5_PNA_alledges",
        "extra": {
            "NeuralNetwork": {
                "Architecture": {
                    "mpnn_type": "PNA",
                    "edge_features": [
                        "misorientation_angle",
                        "edge_length",
                        "relpos_crystal_x",
                        "relpos_crystal_y",
                        "relpos_crystal_z",
                        "slip_transfer_mprime",
                        "elastic_mismatch",
                    ],
                }
            },
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "use_edge_cols": [
                    "misorientation_angle",
                    "edge_length",
                    "relpos_crystal_x",
                    "relpos_crystal_y",
                    "relpos_crystal_z",
                    "slip_transfer_mprime",
                    "elastic_mismatch",
                ],
            },
        },
        "note": "PNA backbone + 7 standardized edge features. Default aggregators (mean+min+max+std) and graph_pooling=max from BASE_RECIPE. Tests whether richer edge physics + PNA's multi-aggregator improves over SAGE.",
    },
    {
        "tag": "H1_a5_yield_stress",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_graph_target_source": "yield_stress",
                "per_graph_transform": "zscore",
            }
        },
        "note": "Per-graph head predicts yield stress (monotonic_plastic[2]); zscore.",
    },
    {
        "tag": "H2_a5_cyc_plastic_strain",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_graph_target_source": "cyc_plastic_strain",
                "per_graph_transform": "zscore",
            }
        },
        "note": "Per-graph head predicts cyclic plastic strain amp (Coffin-Manson); zscore.",
    },
    {
        "tag": "H3_a5_cyc_stress_amp",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_graph_target_source": "cyc_stress_amp",
                "per_graph_transform": "zscore",
            }
        },
        "note": "Per-graph head predicts cyclic stress amplitude (Basquin); zscore.",
    },
    {
        "tag": "H4_a5_aniso_E",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_graph_target_source": "aniso_E",
                "per_graph_transform": "yeo_johnson",
            }
        },
        "note": "Per-graph head predicts elastic anisotropy index (Young's); yeo_johnson.",
    },
    {
        "tag": "H5_a5_aniso_G",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_graph_target_source": "aniso_G",
                "per_graph_transform": "yeo_johnson",
            }
        },
        "note": "Per-graph head predicts shear anisotropy index (G); yeo_johnson.",
    },
    {
        "tag": "H6_a5_cyc_low_stress_amp",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_graph_target_source": "cyc_low_stress_amp",
                "per_graph_transform": "zscore",
            }
        },
        "note": "Per-graph head predicts cyclic stress amplitude at low strain (Δε=0.001, pre-yield) per Anto 2026; zscore. CV is small so dataset variability is limited.",
    },
    {
        "tag": "I1_a5_MTL_full_SAGE",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "zscore",
                    },
                ],
            }
        },
        "note": "Full MTL with SAGE backbone: FIP per-grain + FIP per-graph + 7 mech QoIs simultaneously. Same training recipe as C1.",
    },
    {
        "tag": "K1_a5_MTL_with_gumbel",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "gumbel_mu_pergraph",
                        "source": "gumbel_mu_pergraph",
                        "transform": "zscore",
                    },
                    {
                        "name": "gumbel_beta_pergraph",
                        "source": "gumbel_beta_pergraph",
                        "transform": "zscore",
                    },
                ],
            }
        },
        "note": "I1 + 2 direct Gumbel aux heads (μ, β) computed by method of moments from within-SVE per-grain MAX values. Closes §3.6 paper story with direct distribution prediction.",
    },
    {
        "tag": "K2_a5_MTL_with_gumbel_lpscale",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "zscore",
                        "scale": 10000000000.0,
                    },
                    {
                        "name": "gumbel_mu_pergraph",
                        "source": "gumbel_mu_pergraph",
                        "transform": "zscore",
                    },
                    {
                        "name": "gumbel_beta_pergraph",
                        "source": "gumbel_beta_pergraph",
                        "transform": "zscore",
                    },
                ],
            }
        },
        "note": "K1 + cyc_low_plastic_strain pre-multiplied by 1e10 then zscore (vs K1's plain zscore on raw 1e-10 values). Tests whether numerical conditioning at training time fixes the MTL drop (0.85 single-task → 0.40 K1) on this target.",
    },
    {
        "tag": "K3_a5_MTL_gumbel_perbucket",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "yeo_johnson",
                        "scale": 10000000000.0,
                    },
                    {
                        "name": "gumbel_mu_pergraph",
                        "source": "gumbel_mu_pergraph",
                        "transform": "zscore_per_bucket",
                    },
                    {
                        "name": "gumbel_beta_pergraph",
                        "source": "gumbel_beta_pergraph",
                        "transform": "zscore_per_bucket",
                    },
                ],
            }
        },
        "note": "K2 + per-bucket zscore for Gumbel μ, β (vs K1/K2's pooled zscore). Bucket-conditional CV varies 7× across SVE sizes for β — per-bucket should improve β prediction substantially.",
    },
    {
        "tag": "J1_a5_MTL_trimmed_SAGE",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "use_x_blocks": [
                    "schmid12",
                    "volume",
                    "surface_area",
                    "degree",
                    "Iuu",
                    "Ivv",
                    "J_polar",
                    "I1",
                    "voxel_geom",
                ],
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "zscore",
                    },
                ],
            }
        },
        "note": "MTL with trimmed feature set: schmid12 + 6 grain-shape + voxel_geom = 24 dims (vs 32 in I1). Drops quat / perimeter / Iuv / I2 per Q2b permutation importance.",
    },
    {
        "tag": "I2_a5_MTL_full_PNA",
        "extra": {
            "NeuralNetwork": {
                "Architecture": {
                    "mpnn_type": "PNA",
                    "edge_features": [
                        "misorientation_angle",
                        "edge_length",
                        "relpos_crystal_x",
                        "relpos_crystal_y",
                        "relpos_crystal_z",
                        "slip_transfer_mprime",
                        "elastic_mismatch",
                    ],
                }
            },
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "use_edge_cols": [
                    "misorientation_angle",
                    "edge_length",
                    "relpos_crystal_x",
                    "relpos_crystal_y",
                    "relpos_crystal_z",
                    "slip_transfer_mprime",
                    "elastic_mismatch",
                ],
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "zscore",
                    },
                ],
            },
        },
        "note": "Full MTL with PNA backbone + 7 standardized edge features + 7 mech aux QoIs. Final paper headline cell.",
    },
    {
        "tag": "H7_a5_cyc_low_plastic_strain",
        "extra": {
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "per_graph_target_source": "cyc_low_plastic_strain",
                "per_graph_transform": "zscore",
            }
        },
        "note": "Per-graph head predicts cyclic plastic strain amplitude at low strain (Δε=0.001, near-zero plasticity) per Anto 2026. Original yeo_johnson collapsed to all-zero predictions on tiny values (~3e-10); zscore handles the scale cleanly since the range/std ratio is well-defined.",
    },
]
TRACK_M: List[Dict[str, Any]] = [
    {
        "tag": "M1_K1_dropout",
        "extra": {
            "NeuralNetwork": {"Architecture": {"dropout": 0.1}},
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "gumbel_mu_pergraph",
                        "source": "gumbel_mu_pergraph",
                        "transform": "zscore",
                    },
                    {
                        "name": "gumbel_beta_pergraph",
                        "source": "gumbel_beta_pergraph",
                        "transform": "zscore",
                    },
                ],
            },
        },
        "note": "K1 (SAGE MTL: FIP pn + pg + 7 mech + Gumbel μ, β) + encoder dropout p=0.1 for MC-Dropout epistemic UQ. Activated via nn.Dropout after every SAGEConv. At inference time, leave dropout on (model.train()) and do K=30 forward passes per SVE.",
    }
]
TRACK_E_SPLIT_SEEDS = [42, 7, 13, 21, 99]
TRACK_E_INIT_SEEDS = [42, 137, 2024, 7, 13]


def _make_track_b_cell(
    tag: str, beta_spec_override: Dict[str, Any], note: str
) -> Dict[str, Any]:
    """Clone M1_K1_dropout with only the gumbel_beta_pergraph aux spec changed."""
    base_aux = [
        {"name": "yield_stress", "source": "yield_stress", "transform": "zscore"},
        {
            "name": "cyc_plastic_strain",
            "source": "cyc_plastic_strain",
            "transform": "zscore",
        },
        {"name": "cyc_stress_amp", "source": "cyc_stress_amp", "transform": "zscore"},
        {"name": "aniso_E", "source": "aniso_E", "transform": "yeo_johnson"},
        {"name": "aniso_G", "source": "aniso_G", "transform": "yeo_johnson"},
        {
            "name": "cyc_low_stress_amp",
            "source": "cyc_low_stress_amp",
            "transform": "zscore",
        },
        {
            "name": "cyc_low_plastic_strain",
            "source": "cyc_low_plastic_strain",
            "transform": "zscore",
        },
        {
            "name": "gumbel_mu_pergraph",
            "source": "gumbel_mu_pergraph",
            "transform": "zscore",
        },
    ]
    beta_spec = {"name": "gumbel_beta_pergraph", "source": "gumbel_beta_pergraph"}
    beta_spec.update(beta_spec_override)
    return {
        "tag": tag,
        "extra": {
            "NeuralNetwork": {"Architecture": {"dropout": 0.1}},
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "aux_pergraph_targets": base_aux + [beta_spec],
            },
        },
        "note": note,
    }


TRACK_B_BETA: List[Dict[str, Any]] = [
    _make_track_b_cell(
        tag="B1_logbeta",
        beta_spec_override={"transform": "log"},
        note="B1: log(β) reparameterisation. β is positive ~10^-4; log + zscore may recover learnable signal lost to compressed scale.",
    ),
    _make_track_b_cell(
        tag="B2_yeojohnson_beta",
        beta_spec_override={"transform": "yeo_johnson"},
        note="B2: Yeo–Johnson transform on β (same machinery used for aniso_E/G aux targets).",
    ),
    _make_track_b_cell(
        tag="B3_perbucket_beta",
        beta_spec_override={"transform": "zscore_per_bucket"},
        note="B3: per-bucket z-score on β. Decouples bucket-conditional centre from within-bucket residual.",
    ),
    _make_track_b_cell(
        tag="B5_w3_beta",
        beta_spec_override={"transform": "zscore", "loss_weight": 1.7320508075688772},
        note="B5: post-zscore loss_weight = √3 on β; effective β-MSE upweighting = 3× relative to other aux columns. Tests whether the head needs more gradient pressure rather than better conditioning.",
    ),
]
TRACK_BC_BETA: List[Dict[str, Any]] = [
    _make_track_b_cell(
        tag="BC1_logbeta_w3",
        beta_spec_override={"transform": "log", "loss_weight": 1.7320508075688772},
        note="1.C: B1 + B5 — logβ reparam combined with √3 loss-weight.",
    ),
    _make_track_b_cell(
        tag="BC2_w5_beta",
        beta_spec_override={"transform": "zscore", "loss_weight": 2.23606797749979},
        note="1.C: heavier loss reweighting (√5 ≈ 2.24, β-MSE × 5) — pushes the gradient pressure further on the same setup that B5 used.",
    ),
]
TRACK_NOBETA: List[Dict[str, Any]] = [
    {
        "tag": "M1_K1_dropout_nobeta",
        "extra": {
            "NeuralNetwork": {"Architecture": {"dropout": 0.1}},
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": "mse_stdmatch_75",
                "aux_pergraph_targets": [
                    {
                        "name": "yield_stress",
                        "source": "yield_stress",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_plastic_strain",
                        "source": "cyc_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_stress_amp",
                        "source": "cyc_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "aniso_E",
                        "source": "aniso_E",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "aniso_G",
                        "source": "aniso_G",
                        "transform": "yeo_johnson",
                    },
                    {
                        "name": "cyc_low_stress_amp",
                        "source": "cyc_low_stress_amp",
                        "transform": "zscore",
                    },
                    {
                        "name": "cyc_low_plastic_strain",
                        "source": "cyc_low_plastic_strain",
                        "transform": "zscore",
                    },
                    {
                        "name": "gumbel_mu_pergraph",
                        "source": "gumbel_mu_pergraph",
                        "transform": "zscore",
                    },
                ],
            },
        },
        "note": "Chosen-cell M1 minus the β aux head. Eight aux outputs: 7 mech QoIs + within-SVE µ. β replaced at inference by a training-set σ_pop estimate.",
    }
]


def _make_track_tm_cell(tag: str, loss_recipe: str, note: str) -> Dict[str, Any]:
    """Clone M1_K1_dropout with only `loss_recipe` overridden."""
    base_aux = [
        {"name": "yield_stress", "source": "yield_stress", "transform": "zscore"},
        {
            "name": "cyc_plastic_strain",
            "source": "cyc_plastic_strain",
            "transform": "zscore",
        },
        {"name": "cyc_stress_amp", "source": "cyc_stress_amp", "transform": "zscore"},
        {"name": "aniso_E", "source": "aniso_E", "transform": "yeo_johnson"},
        {"name": "aniso_G", "source": "aniso_G", "transform": "yeo_johnson"},
        {
            "name": "cyc_low_stress_amp",
            "source": "cyc_low_stress_amp",
            "transform": "zscore",
        },
        {
            "name": "cyc_low_plastic_strain",
            "source": "cyc_low_plastic_strain",
            "transform": "zscore",
        },
        {
            "name": "gumbel_mu_pergraph",
            "source": "gumbel_mu_pergraph",
            "transform": "zscore",
        },
        {
            "name": "gumbel_beta_pergraph",
            "source": "gumbel_beta_pergraph",
            "transform": "zscore",
        },
    ]
    return {
        "tag": tag,
        "extra": {
            "NeuralNetwork": {"Architecture": {"dropout": 0.1}},
            "Recipe": {
                "use_per_graph_target": True,
                "graph_attr_source": "size_bucket_onehot",
                "loss_recipe": loss_recipe,
                "aux_pergraph_targets": base_aux,
            },
        },
        "note": note,
    }


TRACK_TM: List[Dict[str, Any]] = [
    _make_track_tm_cell(
        tag="TM1_phtk_05_15",
        loss_recipe="peak_huber_topk_05_15",
        note="peak_huber_topk alpha=0.5 gamma=0.15 beta=0.35 - anik production winner; mild top-decile upweight on top of huber + pairwise rank.",
    ),
    _make_track_tm_cell(
        tag="TM2_phtk_05_30",
        loss_recipe="peak_huber_topk_05_30",
        note="peak_huber_topk alpha=0.5 gamma=0.30 beta=0.20 - more aggressive top-k weight, less rank pressure.",
    ),
    _make_track_tm_cell(
        tag="TM3_pinball_85",
        loss_recipe="pinball_85",
        note="pinball quantile tau=0.85 - mild upper-quantile pull.",
    ),
    _make_track_tm_cell(
        tag="TM4_pinball_95",
        loss_recipe="pinball_95",
        note="pinball quantile tau=0.95 - top-end quantile pull.",
    ),
    _make_track_tm_cell(
        tag="TM5_gumbel_tw_05_30",
        loss_recipe="mse_gumbel_tw_05_30",
        note="Gumbel-CDF tail-weighted MSE alpha=0.5 k=3.0 - smooth Gumbel-percentile tail upweight per Gumbel-MoM fit on within-SVE truth.",
    ),
    _make_track_tm_cell(
        tag="TM6_gumbel_tw_05_50",
        loss_recipe="mse_gumbel_tw_05_50",
        note="Gumbel-CDF tail-weighted MSE alpha=0.5 k=5.0 - more aggressive tail upweight (k controls steepness; k=5 vs k=3 lifts tail weight ~10x).",
    ),
]


def make_track_d(loss_label: str, loss_value: str) -> List[Dict[str, Any]]:
    """Build the 4-cell disambiguation grid for a given loss recipe."""
    base_aux = [
        {"name": "yield_stress", "source": "yield_stress", "transform": "zscore"},
        {
            "name": "cyc_plastic_strain",
            "source": "cyc_plastic_strain",
            "transform": "zscore",
        },
        {"name": "cyc_stress_amp", "source": "cyc_stress_amp", "transform": "zscore"},
        {"name": "aniso_E", "source": "aniso_E", "transform": "yeo_johnson"},
        {"name": "aniso_G", "source": "aniso_G", "transform": "yeo_johnson"},
        {
            "name": "cyc_low_stress_amp",
            "source": "cyc_low_stress_amp",
            "transform": "zscore",
        },
        {
            "name": "cyc_low_plastic_strain",
            "source": "cyc_low_plastic_strain",
            "transform": "zscore",
        },
        {
            "name": "gumbel_mu_pergraph",
            "source": "gumbel_mu_pergraph",
            "transform": "zscore",
        },
        {
            "name": "gumbel_beta_pergraph",
            "source": "gumbel_beta_pergraph",
            "transform": "zscore",
        },
    ]
    out: List[Dict[str, Any]] = []
    for dp in [0.0, 0.1]:
        for has_mtl in [False, True]:
            drop_lbl = "drop" if dp > 0 else "nodrop"
            mtl_lbl = "mtl" if has_mtl else "nomtl"
            tag = f"D_{loss_label}_{drop_lbl}_{mtl_lbl}"
            out.append(
                {
                    "tag": tag,
                    "extra": {
                        "NeuralNetwork": {"Architecture": {"dropout": dp}},
                        "Recipe": {
                            "use_per_graph_target": True,
                            "graph_attr_source": "size_bucket_onehot",
                            "loss_recipe": loss_value,
                            "aux_pergraph_targets": base_aux if has_mtl else None,
                        },
                    },
                    "note": f"Phase A2 disambiguation: dropout={dp}, mtl={has_mtl}, loss={loss_value}. Pins the dropout/MTL/loss contributions to the S1(i) 0.554 -> 0.591 lift.",
                }
            )
    return out


import os as _os

_PNA_V4_DATA_PATH = _os.environ.get(
    "FIP_DATA_PATH",
    _os.environ.get(
        "FIP_V4_DATA_PATH",
        str(release_root() / "examples/cpfem_fip_data/outputs/full_data_v5.pt"),
    ),
)
_PNA_V4_EDGE_7 = [
    "misorientation_angle",
    "edge_length",
    "relpos_crystal_x",
    "relpos_crystal_y",
    "relpos_crystal_z",
    "slip_transfer_mprime",
    "elastic_mismatch",
]
_PNA_V4_AUX_18 = [
    {"name": "yield_stress", "source": "yield_stress", "transform": "zscore"},
    {
        "name": "cyc_plastic_strain",
        "source": "cyc_plastic_strain",
        "transform": "zscore",
    },
    {"name": "cyc_stress_amp", "source": "cyc_stress_amp", "transform": "zscore"},
    {"name": "aniso_E", "source": "aniso_E", "transform": "yeo_johnson"},
    {"name": "aniso_G", "source": "aniso_G", "transform": "yeo_johnson"},
    {
        "name": "cyc_low_stress_amp",
        "source": "cyc_low_stress_amp",
        "transform": "zscore",
    },
    {
        "name": "cyc_low_plastic_strain",
        "source": "cyc_low_plastic_strain",
        "transform": "zscore",
    },
    {
        "name": "lcf_wp_pergraph_max",
        "source": "lcf_wp_pergraph_max",
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_pergraph_max",
        "source": "hcf_fs_pergraph_max",
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_pergraph_max",
        "source": "hcf_wp_pergraph_max",
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_fs_ve_mu_pergraph",
        "source": "lcf_fs_ve_mu_pergraph",
        "transform": "zscore",
    },
    {
        "name": "lcf_fs_ve_sigma_pergraph",
        "source": "lcf_fs_ve_sigma_pergraph",
        "transform": "zscore",
    },
    {
        "name": "lcf_wp_ve_mu_pergraph",
        "source": "lcf_wp_ve_mu_pergraph",
        "transform": "zscore",
    },
    {
        "name": "lcf_wp_ve_sigma_pergraph",
        "source": "lcf_wp_ve_sigma_pergraph",
        "transform": "zscore",
    },
    {
        "name": "hcf_fs_ve_mu_pergraph",
        "source": "hcf_fs_ve_mu_pergraph",
        "transform": "zscore",
    },
    {
        "name": "hcf_fs_ve_sigma_pergraph",
        "source": "hcf_fs_ve_sigma_pergraph",
        "transform": "zscore",
    },
    {
        "name": "hcf_wp_ve_mu_pergraph",
        "source": "hcf_wp_ve_mu_pergraph",
        "transform": "zscore",
    },
    {
        "name": "hcf_wp_ve_sigma_pergraph",
        "source": "hcf_wp_ve_sigma_pergraph",
        "transform": "zscore",
    },
]
_PNA_V4_AUX_12 = _PNA_V4_AUX_18
_PNA_V4_AUX_9 = _PNA_V4_AUX_18
_PNA_V4_AUX_8 = _PNA_V4_AUX_18
_PNA_V4_AUX_7 = _PNA_V4_AUX_18


def _make_pna_v4_cell(
    tag: str,
    *,
    recipe_overrides: Dict[str, Any] = None,
    arch_overrides: Dict[str, Any] = None,
    train_overrides: Dict[str, Any] = None,
    note: str = "",
) -> Dict[str, Any]:
    """Builds a PNA-v4 cell with the standard hyperparameters; per-cell
    overrides via the *_overrides dicts."""
    arch = {
        "mpnn_type": "PNA",
        "hidden_dim": 48,
        "num_conv_layers": 6,
        "activation_function": "elu",
        "dropout": 0.121,
        "edge_features": list(_PNA_V4_EDGE_7),
    }
    if arch_overrides:
        arch.update(arch_overrides)
    train = {
        "batch_size": 64,
        "Optimizer": {"learning_rate": 0.0001},
        "weight_decay": 5.3e-05,
    }
    if train_overrides:
        train.update(train_overrides)
    recipe = {
        "pt_path": _PNA_V4_DATA_PATH,
        "use_per_graph_target": True,
        "graph_attr_source": "size_bucket_onehot",
        "loss_recipe": "peak_huber_topk_05_30",
        "use_edge_cols": list(_PNA_V4_EDGE_7),
        "aux_pergraph_targets": list(_PNA_V4_AUX_8),
    }
    if recipe_overrides:
        recipe.update(recipe_overrides)
    return {
        "tag": tag,
        "extra": {
            "NeuralNetwork": {"Architecture": arch, "Training": train},
            "Recipe": recipe,
        },
        "note": note,
    }


_PNA_V4_EDGE_XFORMS = [
    "inv_edge_length",
    "edge_length_rbf_2um",
    "cos_misor",
    "sin_misor",
]
_PNA_V4_EDGE_11 = list(_PNA_V4_EDGE_7) + list(_PNA_V4_EDGE_XFORMS)
_PNA_V4_X_BLOCKS_DEFAULT = [
    "quat",
    "schmid",
    "volume",
    "surface_area",
    "perimeter",
    "degree",
    "Iuu",
    "Ivv",
    "Iuv",
    "J_polar",
    "I1",
    "I2",
]
_PNA_V4_X_BLOCKS_ENRICHED = _PNA_V4_X_BLOCKS_DEFAULT + [
    "voxel_geom",
    "schmid12",
    "log_volume",
    "inv_volume",
    "surface_to_vol",
    "schmid_asymmetry",
    "schmid_std",
    "sve_boundary",
]
_PNA_V4_X_BLOCKS_CROSSVAR = _PNA_V4_X_BLOCKS_ENRICHED + [
    "hcf_fs_fip_dist",
    "lcf_wp_fip_dist",
    "hcf_wp_fip_dist",
]
TRACK_PNA_V4: List[Dict[str, Any]] = [
    _make_pna_v4_cell(
        tag="P0_pna_baseline_v4",
        note="Phase 2 P0: PNA + 7 edges baseline on v4. trial 6 hyperparams, 8-aux MTL, peak_huber_topk_05_30 loss, batch=64. Reference for all later PNA-v4 phases.",
    ),
    _make_pna_v4_cell(
        tag="P1_pna_edgexform",
        arch_overrides={"edge_features": list(_PNA_V4_EDGE_11)},
        recipe_overrides={"use_edge_cols": list(_PNA_V4_EDGE_11)},
        note="Phase 3 P1: P0 + 4 physics-aware edge transforms (1/edge_length, RBF[2um], cos+sin misor). edge_dim=11. Tests proximity-prior + low/high-angle GB physics.",
    ),
    _make_pna_v4_cell(
        tag="P2_pna_nodexform",
        arch_overrides={"edge_features": list(_PNA_V4_EDGE_11)},
        recipe_overrides={
            "use_edge_cols": list(_PNA_V4_EDGE_11),
            "use_x_blocks": list(_PNA_V4_X_BLOCKS_ENRICHED),
        },
        note="Phase 4 P2: P1 + enhanced node features. node_dim=38: default 15 + voxel_geom + schmid12 + 5 derived (log_volume, inv_volume, surface_to_vol, schmid_asymmetry, schmid_std). Tests rich microstructure input.",
    ),
    _make_pna_v4_cell(
        tag="P3_pna_crossvariant",
        arch_overrides={"edge_features": list(_PNA_V4_EDGE_11)},
        recipe_overrides={
            "use_edge_cols": list(_PNA_V4_EDGE_11),
            "use_x_blocks": list(_PNA_V4_X_BLOCKS_CROSSVAR),
        },
        note="Phase 5 P3 (the v4 thesis test): P2 + cross-variant within-grain features (HCF/FS, LCF/WP, HCF/WP — 10 cols each = 30). node_dim=68. Drops 58 NaN-HCF graphs (998 -> 940). Target: pg_R^2 >= 0.65.",
    ),
    _make_pna_v4_cell(
        tag="P6_pna_kitchen_sink",
        arch_overrides={"edge_features": list(_PNA_V4_EDGE_11)},
        recipe_overrides={
            "use_edge_cols": list(_PNA_V4_EDGE_11),
            "use_x_blocks": list(_PNA_V4_X_BLOCKS_ENRICHED),
            "per_grain_aux_targets": [
                {
                    "name": "fs_within_mean",
                    "field": "lcf_fs_fip_dist",
                    "col": 1,
                    "transform": "yeo_johnson",
                },
                {
                    "name": "fs_within_std",
                    "field": "lcf_fs_fip_dist",
                    "col": 2,
                    "transform": "yeo_johnson",
                },
                {
                    "name": "fs_within_p99",
                    "field": "lcf_fs_fip_dist",
                    "col": 4,
                    "transform": "yeo_johnson",
                },
                {
                    "name": "fs_within_lognorm_mu",
                    "field": "lcf_fs_fip_dist",
                    "col": 6,
                    "transform": "zscore",
                },
                {
                    "name": "fs_within_lognorm_sigma",
                    "field": "lcf_fs_fip_dist",
                    "col": 7,
                    "transform": "zscore",
                },
                {
                    "name": "hcf_fs_max",
                    "field": "hcf_fs_fip_dist",
                    "col": 0,
                    "transform": "yeo_johnson",
                },
                {
                    "name": "lcf_wp_max",
                    "field": "lcf_wp_fip_dist",
                    "col": 0,
                    "transform": "yeo_johnson",
                },
                {
                    "name": "hcf_wp_max",
                    "field": "hcf_wp_fip_dist",
                    "col": 0,
                    "transform": "yeo_johnson",
                },
            ],
        },
        note="Kitchen-sink: P1 edges + P2 enriched node features + always-on residuals + 8 per-grain aux targets covering (a) within-grain LCF/FS distribution shape and (b) cross-variant per-grain MAXes. NaN-HCF graphs dropped (998 -> 940); per-grain NaN cells excluded from metrics + parity at eval time via per_grain_aux_nan_mask.",
    ),
]
PROMOTE_EXTRA_SEEDS = [137, 2024]
PROMOTE_THRESHOLDS = {
    "top10_slope_min": 0.18,
    "pn_r2_min": 0.55,
    "pgmax_pearson_min": 0.65,
}
_V9_PERGRAPH_TARGETS_14 = [
    {
        "name": "lcf_fs_pergraph_max",
        "source": "lcf_fs_pergraph_max",
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_pergraph_max",
        "source": "lcf_wp_pergraph_max",
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_pergraph_max",
        "source": "hcf_fs_pergraph_max",
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_pergraph_max",
        "source": "hcf_wp_pergraph_max",
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_fs_ve_mu_pergraph",
        "source": "lcf_fs_ve_mu_pergraph",
        "transform": "zscore",
    },
    {
        "name": "lcf_wp_ve_mu_pergraph",
        "source": "lcf_wp_ve_mu_pergraph",
        "transform": "zscore",
    },
    {
        "name": "hcf_fs_ve_mu_pergraph",
        "source": "hcf_fs_ve_mu_pergraph",
        "transform": "zscore",
    },
    {
        "name": "hcf_wp_ve_mu_pergraph",
        "source": "hcf_wp_ve_mu_pergraph",
        "transform": "zscore",
    },
    {"name": "yield_stress", "source": "yield_stress", "transform": "zscore"},
    {"name": "hardening_modulus", "source": "hardening_modulus", "transform": "zscore"},
    {"name": "Wp_lcf", "source": "Wp_lcf", "transform": "zscore"},
    {"name": "Wp_hcf", "source": "Wp_hcf", "transform": "zscore"},
    {"name": "sigma_amp_lcf", "source": "sigma_amp_lcf", "transform": "zscore"},
    {"name": "sigma_amp_hcf", "source": "sigma_amp_hcf", "transform": "zscore"},
]
_V9_PERGRAPH_TARGETS_13 = _V9_PERGRAPH_TARGETS_14
_V8_PERGRAPH_QOIS_12 = _V9_PERGRAPH_TARGETS_14[1:]
_V7_AUX_PG_14 = _V8_PERGRAPH_QOIS_12
_V7_AUX_PG_12 = _V8_PERGRAPH_QOIS_12
_V9_PERGRAIN_TARGETS_8 = [
    {
        "name": "lcf_fs_max",
        "field": "lcf_fs_fip_dist",
        "col": 0,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_max",
        "field": "lcf_wp_fip_dist",
        "col": 0,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_max",
        "field": "hcf_fs_fip_dist",
        "col": 0,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_max",
        "field": "hcf_wp_fip_dist",
        "col": 0,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_fs_within_mean",
        "field": "lcf_fs_fip_dist",
        "col": 1,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_within_mean",
        "field": "lcf_wp_fip_dist",
        "col": 1,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_within_mean",
        "field": "hcf_fs_fip_dist",
        "col": 1,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_within_mean",
        "field": "hcf_wp_fip_dist",
        "col": 1,
        "transform": "yeo_johnson",
    },
]
_V9b_PERGRAIN_DISTSHAPE_A_12 = list(_V9_PERGRAIN_TARGETS_8) + [
    {
        "name": "lcf_fs_logsigma_voxel",
        "field": "lcf_fs_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_logsigma_voxel",
        "field": "lcf_wp_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_logsigma_voxel",
        "field": "hcf_fs_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_logsigma_voxel",
        "field": "hcf_wp_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
]
_V9b_PERGRAIN_DISTSHAPE_B_16 = list(_V9_PERGRAIN_TARGETS_8) + [
    {
        "name": "lcf_fs_logmu_voxel",
        "field": "lcf_fs_fip_dist",
        "col": 6,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_logmu_voxel",
        "field": "lcf_wp_fip_dist",
        "col": 6,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_logmu_voxel",
        "field": "hcf_fs_fip_dist",
        "col": 6,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_logmu_voxel",
        "field": "hcf_wp_fip_dist",
        "col": 6,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_fs_logsigma_voxel",
        "field": "lcf_fs_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_logsigma_voxel",
        "field": "lcf_wp_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_logsigma_voxel",
        "field": "hcf_fs_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_logsigma_voxel",
        "field": "hcf_wp_fip_dist",
        "col": 7,
        "transform": "yeo_johnson",
    },
]
_V9b_PERGRAIN_DISTSHAPE_C_16 = list(_V9_PERGRAIN_TARGETS_8) + [
    {
        "name": "lcf_fs_p90",
        "field": "lcf_fs_fip_dist",
        "col": 3,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_p90",
        "field": "lcf_wp_fip_dist",
        "col": 3,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_p90",
        "field": "hcf_fs_fip_dist",
        "col": 3,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_p90",
        "field": "hcf_wp_fip_dist",
        "col": 3,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_fs_p99",
        "field": "lcf_fs_fip_dist",
        "col": 4,
        "transform": "yeo_johnson",
    },
    {
        "name": "lcf_wp_p99",
        "field": "lcf_wp_fip_dist",
        "col": 4,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_fs_p99",
        "field": "hcf_fs_fip_dist",
        "col": 4,
        "transform": "yeo_johnson",
    },
    {
        "name": "hcf_wp_p99",
        "field": "hcf_wp_fip_dist",
        "col": 4,
        "transform": "yeo_johnson",
    },
]
_V9_PERGRAIN_QOIS_7 = _V9_PERGRAIN_TARGETS_8[1:]
_V8_PERGRAIN_QOIS_15 = _V9_PERGRAIN_QOIS_7
_V7_AUX_PN_15 = _V9_PERGRAIN_QOIS_7
_V7_AUX_PN_6 = _V9_PERGRAIN_QOIS_7


def make_v7_kitchen_sage(
    extra_recipe: Dict[str, Any] = None,
    extra_arch: Dict[str, Any] = None,
    extra_train: Dict[str, Any] = None,
    tag: str = "P9_sage_kitchen_v7",
) -> Dict[str, Any]:
    """Build the v7 SAGE kitchen-sink cell with σ-trimmed aux.

    Returns a (tag, extra) dict consumable by run_cell(). HPO knobs go in
    the `extra_*` overrides; everything not overridden uses the locked v7
    defaults below.
    """
    arch = {
        "mpnn_type": "SAGE",
        "hidden_dim": 48,
        "num_conv_layers": 6,
        "activation_function": "elu",
        "dropout": 0.121,
        "edge_features": [],
        "graph_pooling": "max",
    }
    if extra_arch:
        arch.update(extra_arch)
    train = {
        "batch_size": 16,
        "Optimizer": {"learning_rate": 0.0001},
        "weight_decay": 5.3e-05,
    }
    if extra_train:
        train.update(extra_train)
    recipe = {
        "pt_path": _PNA_V4_DATA_PATH,
        "use_per_graph_target": True,
        "graph_attr_source": "size_bucket_onehot",
        "graph_attr_conditioning_mode": "concat_node",
        "loss_recipe": "peak_huber_topk_05_30",
        "use_edge_cols": [],
        "use_x_blocks": list(_PNA_V4_X_BLOCKS_ENRICHED),
        "aux_pergraph_targets": list(_V8_PERGRAPH_QOIS_12),
        "per_grain_aux_targets": list(_V9_PERGRAIN_QOIS_7),
    }
    if extra_recipe:
        recipe.update(extra_recipe)
    return {
        "tag": tag,
        "extra": {
            "NeuralNetwork": {"Architecture": arch, "Training": train},
            "Recipe": recipe,
        },
        "note": "v7 SAGE kitchen-sink: 14 per-graph aux + 6 per-grain aux = 20 multi-task outputs (σ targets dropped per P8 result).",
    }


TRACK_V7: List[Dict[str, Any]] = [make_v7_kitchen_sage()]
WALL_PER_CELL_HOURS = {"A": 0.2, "L": 0.2, "promote": 0.2, "E": 0.2}
