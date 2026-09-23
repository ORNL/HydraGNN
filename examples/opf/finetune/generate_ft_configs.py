"""Generate fine-tuning config JSON files for FT1–FT4 strategies.

Run from the finetune/ directory:
    python generate_ft_configs.py

FT1 is a graph-level binary feasibility classification task:
  - Feasible samples: taken from an existing OPF dataset (label = 1.0)
  - Infeasible samples: synthesised by overloading load features (label = 0.0)
  - Preprocessing: run generate_infeasible_samples.py before training FT1
  - Loss: binary_cross_entropy_with_logits (BCE)
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EDGE_TYPES = [
    {"source_type": "bus", "relation": "ac_line", "target_type": "bus", "dim": 9},
    {"source_type": "bus", "relation": "transformer", "target_type": "bus", "dim": 11},
    {
        "source_type": "generator",
        "relation": "generator_link",
        "target_type": "bus",
        "dim": 0,
    },
    {
        "source_type": "bus",
        "relation": "generator_link",
        "target_type": "generator",
        "dim": 0,
    },
    {"source_type": "load", "relation": "load_link", "target_type": "bus", "dim": 0},
    {"source_type": "bus", "relation": "load_link", "target_type": "load", "dim": 0},
    {"source_type": "shunt", "relation": "shunt_link", "target_type": "bus", "dim": 0},
    {"source_type": "bus", "relation": "shunt_link", "target_type": "shunt", "dim": 0},
]


def _edge_types_for(mpnn_type):
    if mpnn_type in {"HeteroGIN", "HeteroHGT", "HeteroSAGE"}:
        return [{**edge_type, "dim": 0} for edge_type in EDGE_TYPES]
    return [edge_type.copy() for edge_type in EDGE_TYPES]


def _base_training(lr, num_epoch, regime):
    return {
        "num_epoch": num_epoch,
        "batch_size": 32,
        "patience": 15,
        "early_stopping": True,
        "Checkpoint": True,
        "checkpoint_warmup": 2,
        "continue": 0,
        "startfrom": "existing_model",
        "DomainLoss": {
            "enabled": False,
            "smoothness_weight": 0.001,
            "transformer_smoothness_weight": 0.001,
            "voltage_bound_weight": 0.01,
            "voltage_bound_feature_indices": [2, 3],
            "voltage_output_index": -1,
        },
        "Optimizer": {"type": "AdamW", "learning_rate": lr},
        "conv_checkpointing": False,
        "loss_function_type": "mse",
        "precision": "fp32",
        "_ft_regime": regime,
    }


def _base_arch(mpnn_type, hd, nl, freeze_conv, node_target_type, out_dim):
    return {
        "mpnn_type": mpnn_type,
        "hidden_dim": hd,
        "num_conv_layers": nl,
        "pe_dim": 0,
        "max_neighbours": 100,
        "hetero_attention_heads": 4,
        "edge_types": _edge_types_for(mpnn_type),
        "node_input_dims": {"bus": 4, "generator": 11, "load": 2, "shunt": 2},
        "output_heads": {
            "node": [
                {
                    "type": "branch-0",
                    "architecture": {
                        "num_headlayers": 2,
                        "dim_headlayers": [32, 16],
                        "type": "mlp",
                    },
                }
            ]
        },
        "task_weights": [1.0],
        "hetero_pooling_mode": "sum",
        "node_target_type": node_target_type,
        "global_attn_engine": None,
        "global_attn_type": None,
        "global_attn_heads": 0,
        "output_dim": [out_dim],
        "output_type": ["node"],
        "num_nodes": 31,
        "input_dim": 4,
        "pna_deg": None,
        "activation_function": "relu",
        "hetero_attention_negative_slope": 0.2,
        "hetero_edge_type_emb_dim": 16,
        "hetero_edge_attr_emb_dim": 16,
        "SyncBatchNorm": False,
        "freeze_conv_layers": freeze_conv,
    }


def _base_arch_graph(mpnn_type, hd, nl, freeze_conv):
    """Architecture config for graph-level binary classification (FT1)."""
    return {
        "mpnn_type": mpnn_type,
        "hidden_dim": hd,
        "num_conv_layers": nl,
        "pe_dim": 0,
        "max_neighbours": 100,
        "hetero_attention_heads": 4,
        "edge_types": _edge_types_for(mpnn_type),
        "node_input_dims": {"bus": 4, "generator": 11, "load": 2, "shunt": 2},
        "output_heads": {
            "graph": [
                {
                    "type": "branch-0",
                    "architecture": {
                        "num_sharedlayers": 1,
                        "dim_sharedlayers": 64,
                        "num_headlayers": 2,
                        "dim_headlayers": [32, 16],
                    },
                }
            ]
        },
        "task_weights": [1.0],
        "hetero_pooling_mode": "sum",
        "global_attn_engine": None,
        "global_attn_type": None,
        "global_attn_heads": 0,
        "output_dim": [1],
        "output_type": ["graph"],
        # num_nodes is not fixed for the classification task (variable topology)
        "num_nodes": None,
        "input_dim": 4,
        "pna_deg": None,
        "activation_function": "relu",
        "hetero_attention_negative_slope": 0.2,
        "hetero_edge_type_emb_dim": 16,
        "hetero_edge_attr_emb_dim": 16,
        "SyncBatchNorm": False,
        "freeze_conv_layers": freeze_conv,
    }


HETERO_INPUTS = [
    {"name": "context", "level": "graph", "dim": 1},
    {"name": "bus_base_kv", "level": "node", "dim": 1, "node_type": "bus"},
    {"name": "bus_type", "level": "node", "dim": 1, "node_type": "bus"},
    {"name": "bus_vmin", "level": "node", "dim": 1, "node_type": "bus"},
    {"name": "bus_vmax", "level": "node", "dim": 1, "node_type": "bus"},
    {"name": "gen_mbase", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_pg_mid", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_pmin", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_pmax", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_qg_mid", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_qmin", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_qmax", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_vg_status", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_cost_c2", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_cost_c1", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "gen_cost_c0", "level": "node", "dim": 1, "node_type": "generator"},
    {"name": "load_pd", "level": "node", "dim": 1, "node_type": "load"},
    {"name": "load_qd", "level": "node", "dim": 1, "node_type": "load"},
    {"name": "shunt_bs", "level": "node", "dim": 1, "node_type": "shunt"},
    {"name": "shunt_gs", "level": "node", "dim": 1, "node_type": "shunt"},
]


def _variables(output_name, level, dim, node_type=None):
    output = {"name": output_name, "level": level, "dim": dim}
    if node_type is not None:
        output["node_type"] = node_type
    return {
        "graph_type": "heterogeneous",
        "node_types": ["bus", "generator", "load", "shunt"],
        "inputs": HETERO_INPUTS,
        "outputs": [output],
    }


# Best HPO hyperparameters from Table VII of the manuscript
ARCHS = {
    "HeteroSAGE": {"hd": 141, "nl": 5},
    "HeteroHEAT": {"hd": 232, "nl": 6},
}

# Learning rates per regime:
#   head_only -> higher LR (only head updates; encoder frozen)
#   partial   -> medium LR (last conv + head)
#   full      -> lower LR  (preserve pretrained representations)
REGIME_LR = {
    "head_only": 1.0e-3,
    "partial": 5.0e-4,
    "full": 1.0e-4,
}

STRATEGIES = {
    "FT1_topology": {
        "desc": (
            "Topology-specific fine-tuning: pretrain on 10-case corpus, "
            "fine-tune on held-out topology pglib_opf_case118_ieee."
        ),
        "case": "pglib_opf_case118_ieee",
        "groups": "20",
        "max_samples": None,
        "topo_perturb": False,
        "epochs": 50,
        "target": "bus",
        "data_modelname": "FT1_topology_data",
    },
    "FT1_feasibility_classification": {
        "desc": (
            "Feasibility classification: binary graph-level prediction of "
            "whether an OPF instance is feasible (1) or infeasible (0). "
            "Infeasible samples are synthesised by scaling load features by "
            "an overload factor so that total demand exceeds generation capacity. "
            "Preprocessing: run generate_infeasible_samples.py to create the "
            "mixed dataset before training."
        ),
        # overload_factor is metadata only; actual value set in generate script
        "overload_factor": 6.0,
        "epochs": 50,
        "target": "graph",
    },
    "FT2_operating_condition": {
        "desc": (
            "Operating-condition fine-tuning: fine-tune on a new distribution "
            "of load/generation profiles for pglib_opf_case14_ieee with a "
            "limited labeled budget (5000 samples)."
        ),
        "case": "pglib_opf_case14_ieee",
        "groups": "2",
        "max_samples": 5000,
        "topo_perturb": False,
        "epochs": 50,
        "target": "bus",
    },
    "FT3_contingency": {
        "desc": (
            "Contingency fine-tuning: adapt to N-1 topological perturbations "
            "(topological_perturbations=True) for pglib_opf_case118_ieee."
        ),
        "case": "pglib_opf_case118_ieee",
        "groups": "20",
        "max_samples": None,
        "topo_perturb": True,
        "epochs": 50,
        "target": "bus",
    },
    "FT4_task_specific": {
        "desc": (
            "Task-specific fine-tuning: adapt the pretrained bus-voltage encoder "
            "to generator dispatch prediction (node_target_type=generator)."
        ),
        "case": "pglib_opf_case118_ieee",
        "groups": "20",
        "max_samples": None,
        "topo_perturb": False,
        "epochs": 50,
        "target": "generator",
    },
}


def _base_training_classify(lr, num_epoch, regime):
    """Training config for binary classification tasks (FT1)."""
    return {
        "num_epoch": num_epoch,
        "batch_size": 32,
        "patience": 15,
        "early_stopping": True,
        "Checkpoint": True,
        "checkpoint_warmup": 2,
        "continue": 0,
        "startfrom": "existing_model",
        "Optimizer": {"type": "AdamW", "learning_rate": lr},
        "conv_checkpointing": False,
        "loss_function_type": "binary_cross_entropy",
        "precision": "fp32",
        "_ft_regime": regime,
    }


def generate_all():
    for ft_dir, fm in STRATEGIES.items():
        out_dir = os.path.join(SCRIPT_DIR, ft_dir)
        os.makedirs(out_dir, exist_ok=True)

        tgt = fm["target"]

        for arch_name, ap in ARCHS.items():
            for regime, freeze_conv in [
                ("head_only", True),
                ("partial", False),
                ("full", False),
            ]:
                lr = REGIME_LR[regime]

                if tgt == "graph":
                    # FT1: graph-level binary classification
                    arch = _base_arch_graph(arch_name, ap["hd"], ap["nl"], freeze_conv)
                    training = _base_training_classify(lr, fm["epochs"], regime)
                    variables = _variables("feasibility", "graph", 1)
                    # Shared dataset (arch-independent)
                    data_modelname = "FT1_feasibility_data"
                    cfg = {
                        "_ft_strategy": ft_dir,
                        "_ft_description": fm["desc"],
                        "_ft_overload_factor": fm["overload_factor"],
                        "ft_data_modelname": data_modelname,
                        "Verbosity": {"level": 2},
                        "NeuralNetwork": {
                            "Architecture": arch,
                            "Training": training,
                        },
                        "Variables": variables,
                        "Visualization": {
                            "plot_init_solution": False,
                            "plot_hist_solution": False,
                            "create_plots": False,
                        },
                    }
                else:
                    # FT2 / FT3 / FT4: node-level regression
                    out_dim = 2  # bus [Va, Vm] or generator [Pg, Qg]
                    node_target_type = tgt  # "bus" or "generator"
                    output_name = (
                        "generator_pg_qg" if tgt == "generator" else "bus_va_vm"
                    )
                    variables = _variables(
                        output_name, "node", out_dim, node_target_type
                    )
                    arch = _base_arch(
                        arch_name,
                        ap["hd"],
                        ap["nl"],
                        freeze_conv,
                        node_target_type,
                        out_dim,
                    )
                    training = _base_training(lr, fm["epochs"], regime)
                    cfg = {
                        "_ft_strategy": ft_dir,
                        "_ft_description": fm["desc"],
                        "_ft_case_name": fm["case"],
                        "_ft_num_groups": fm["groups"],
                        "_ft_max_samples": fm["max_samples"],
                        "_ft_topological_perturbations": fm["topo_perturb"],
                        "ft_data_modelname": fm.get(
                            "data_modelname", f"{ft_dir}_{arch_name}_data"
                        ),
                        "Verbosity": {"level": 2},
                        "NeuralNetwork": {
                            "Architecture": arch,
                            "Training": training,
                        },
                        "Variables": variables,
                        "Visualization": {
                            "plot_init_solution": False,
                            "plot_hist_solution": False,
                            "create_plots": False,
                        },
                    }

                fname = os.path.join(out_dir, f"config_{arch_name}_{regime}.json")
                with open(fname, "w") as f:
                    json.dump(cfg, f, indent=4)
                print(f"Created {fname}")

    print("\nAll fine-tuning configs generated.")


if __name__ == "__main__":
    generate_all()
