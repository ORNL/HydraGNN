from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence

import torch

import hydragnn

try:
    from .CP2D import CP2D, XMode, _print_image_split_stats, split_by_sve_images_per_vf, split_by_sve_json
except ImportError:  # pragma: no cover
    from CP2D import CP2D, XMode, _print_image_split_stats, split_by_sve_images_per_vf, split_by_sve_json


def main(
    mpnn_type=None,
    global_attn_engine=None,
    global_attn_type=None,
    x_mode: XMode = "quat",
    log: str = "std_sage4",
    extra_edge_attrs_str: str = "quat_angle",
    vfs: Optional[Sequence[float]] = None,
    split_seed: int = 0,
    task_weights_str: Optional[str] = None,
    loss_function_type: Optional[str] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    num_epoch: Optional[int] = None,
    patience: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    num_conv_layers: Optional[int] = None,
    head_dims_str: Optional[str] = None,
    n_train_images: int = 26,
    n_val_images: int = 2,
    n_test_images: int = 2,
    root: Optional[str] = None,
    graphs_dir: Optional[str] = None,
    y_labels_csv: Optional[str] = None,
    split_json: Optional[str] = None,
):

    random_state = 0
    torch.manual_seed(random_state)

    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.abspath(root or repo_root)
    resolved_graphs_dir = os.path.abspath(graphs_dir or os.path.join(dataset_root, "graphs"))
    resolved_y_labels_csv = (
        os.path.abspath(y_labels_csv) if y_labels_csv else os.path.join(resolved_graphs_dir, "y_labels.csv")
    )

    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pna.json")
    with open(filename, "r") as f:
        config = json.load(f)

    if global_attn_engine:
        config["NeuralNetwork"]["Architecture"]["global_attn_engine"] = global_attn_engine
    if global_attn_type:
        config["NeuralNetwork"]["Architecture"]["global_attn_type"] = global_attn_type
    if mpnn_type:
        config["NeuralNetwork"]["Architecture"]["mpnn_type"] = mpnn_type
    if hidden_dim is not None:
        config["NeuralNetwork"]["Architecture"]["hidden_dim"] = hidden_dim
    if num_conv_layers is not None:
        config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = num_conv_layers
    if head_dims_str:
        head_dims = [int(s.strip()) for s in head_dims_str.split(",") if s.strip()]
        if not head_dims:
            raise ValueError("head_dims must have at least one integer value.")
        config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"]["dim_headlayers"] = head_dims
        config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"]["num_headlayers"] = len(head_dims)

    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    # Vector outputs: stress-strain curves (stress values only) for x/y.
    # Each curve is length 24.
    # stress_len = 10
    stress_len = CP2D.STRESS_STRAIN_LEN
    scalar_labels = [
        "out_C11",
        "out_C12",
        "out_C22",
        "out_C33",
        "K_R",
        "G_R",
        "avg_yield",
        "avg_hard",
        "out_yieldx0.002",
        "out_yieldy0.002",
    ]
    var_config["graph_feature_names"] = [
        "stress_strain_x",
        "stress_strain_y",
        *scalar_labels,
    ]

    if task_weights_str:
        task_weights = [float(s.strip()) for s in task_weights_str.split(",") if s.strip()]
        if len(task_weights) != len(var_config["graph_feature_names"]):
            raise ValueError(
                "task_weights length must match number of outputs "
                f"({len(var_config['graph_feature_names'])})."
            )
        config["NeuralNetwork"]["Architecture"]["task_weights"] = task_weights
    else:
        config["NeuralNetwork"]["Architecture"]["task_weights"] = [
            1.0
        ] * len(var_config["graph_feature_names"])
    var_config["output_names"] = var_config["graph_feature_names"]
    var_config["output_index"] = list(range(len(var_config["graph_feature_names"])))
    var_config["output_dim"] = [
        stress_len if name in ("stress_strain_x", "stress_strain_y") else 1
        for name in var_config["output_names"]
    ]
    var_config["type"] = ["graph"] * len(var_config["output_names"])
    var_config["graph_feature_dims"] = list(var_config["output_dim"])

    var_config["input_node_features"] =[]
    log_name = f"{log}" 
    hydragnn.utils.print.print_utils.setup_log(log_name)
    extra_edge_attrs = [s.strip() for s in extra_edge_attrs_str.split(",") if s.strip() and s.strip().lower() != "none"]

    if extra_edge_attrs is not None:
        edge_features = []
        config["NeuralNetwork"]["Architecture"]["edge_features"] = edge_features
        if "quat_angle" in extra_edge_attrs:
            edge_features += ["quat_angle_mismatch"]
        if "quat_rel" in extra_edge_attrs:
            edge_features += ["quat_rel_w", "quat_rel_x", "quat_rel_y", "quat_rel_z"]
        if "euler_mismatch" in extra_edge_attrs:
            edge_features += ["euler_mismatch"]
        if "phase_pair" in extra_edge_attrs:
            edge_features += ["phase_pair_code"]


    # Dataset load (same style as your CPSAGE usage)
    dataset = CP2D(
        root=dataset_root,
        var_config=var_config,
        force_reload=True,
        scaler="standard",
        x_mode=x_mode,
        graphs_dir=resolved_graphs_dir,
        y_labels_csv=resolved_y_labels_csv,
        y_labels = var_config["graph_feature_names"],
        edge_attr_mode="none",
        extra_edge_attrs=extra_edge_attrs,  # e.g. ["euler_mismatch"] or ["euler_mismatch","quat_angle"]
        shuffle=True,
        shuffle_seed=42,
    )

    
    # Make var_config consistent with what CP2D writes into Data.x
    # HydraGNN will slice Data.x using `input_node_features` and `node_feature_dims`.
    var_config["input_node_features"] = list(range(len(dataset.x_labels)))
    var_config["node_feature_names"] = list(dataset.x_labels)
    var_config["node_feature_dims"] = [1] * len(dataset.x_labels)

    config["NeuralNetwork"]["Variables_of_interest"] = var_config
    if loss_function_type is not None:
        config["NeuralNetwork"]["Training"]["loss_function_type"] = loss_function_type
    if learning_rate is not None:
        config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"] = learning_rate
    if batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = batch_size
    if num_epoch is not None:
        config["NeuralNetwork"]["Training"]["num_epoch"] = num_epoch
    if patience is not None:
        config["NeuralNetwork"]["Training"]["patience"] = patience

    # Stratified split by (domain size, volume fraction) after shuffle.
    # Split by SVE/image id within each volume fraction.
    # This keeps all partitions (all sizes + blocks) from the same image together.
    split_save_path = os.path.join(dataset.processed_dir, f"split_by_sve_seed{split_seed}.json")
    if split_json:
        train, val, test = split_by_sve_json(
            dataset,
            split_json,
            allowed_vfs=vfs,
        )
    else:
        train, val, test = split_by_sve_images_per_vf(
            dataset,
            n_train_images=n_train_images,
            n_val_images=n_val_images,
            n_test_images=n_test_images,
            seed=split_seed,
            allowed_vfs=vfs,
            save_path=split_save_path,
        )
    _print_image_split_stats("train", train)
    _print_image_split_stats("val", val)
    _print_image_split_stats("test", test)

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=config["Visualization"]["create_plots"],
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CP2D training with optional model type.")
    parser.add_argument("--mpnn_type", type=str, default=None)
    parser.add_argument("--global_attn_engine", type=str, default=None)
    parser.add_argument("--global_attn_type", type=str, default=None)
    parser.add_argument("--x_mode", type=str, choices=["euler", "quat"], default="quat")
    parser.add_argument("--log", type=str, default="std_sage4")
    parser.add_argument(
        "--vfs",
        type=str,
        default="",
        help="Comma-separated volume fractions to include (e.g. '0.1,0.45,0.9'). Empty = include all.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Seed used when generating a new SVE-image split (ignored if --split_json is provided).",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="Optional path to an existing split JSON, e.g. splits/split_by_sve_seed0.json.",
    )
    parser.add_argument(
        "--n_train_images",
        type=int,
        default=26,
        help="Number of unique SVE images per vf to use for training.",
    )
    parser.add_argument(
        "--n_val_images",
        type=int,
        default=2,
        help="Number of unique SVE images per vf to use for validation.",
    )
    parser.add_argument(
        "--n_test_images",
        type=int,
        default=2,
        help="Number of unique SVE images per vf to use for testing.",
    )
    parser.add_argument(
        "--extra_edge_attrs",
        type=str,
        default="none",
        help="Comma-separated list: quat_angle, quat_rel, euler_mismatch, phase_pair (e.g. 'quat_rel' or 'quat_angle,quat_rel')",
    )
    parser.add_argument(
        "--task_weights",
        type=str,
        default="",
        help="Comma-separated task weights (one per output head). Example: '1.0,1.0'.",
    )
    parser.add_argument("--loss", type=str, default=None, help="Override loss_function_type (e.g. mse, mae).")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num_epoch", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Override hidden_dim.")
    parser.add_argument("--num_conv_layers", type=int, default=None, help="Override num_conv_layers.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Dataset working directory. Defaults to the `cp2d/` repository root.",
    )
    parser.add_argument(
        "--graphs_dir",
        type=str,
        default=None,
        help="Directory containing graph .pt files. Defaults to <root>/graphs.",
    )
    parser.add_argument(
        "--y_labels_csv",
        type=str,
        default=None,
        help="Optional explicit path to y_labels.csv. Defaults to <graphs_dir>/y_labels.csv.",
    )
    parser.add_argument(
        "--head_dims",
        type=str,
        default="",
        help="Comma-separated head layer dims for graph output head (e.g. '128,64').",
    )
    args = parser.parse_args()
    vfs: Optional[List[float]]
    if args.vfs.strip():
        vfs = [float(s.strip()) for s in args.vfs.split(",") if s.strip()]
    else:
        vfs = None
    main(
        mpnn_type=args.mpnn_type,
        global_attn_engine=args.global_attn_engine,
        global_attn_type=args.global_attn_type,
        x_mode=args.x_mode,
        log=args.log,
        extra_edge_attrs_str=args.extra_edge_attrs,
        vfs=vfs,
        split_seed=args.split_seed,
        n_train_images=args.n_train_images,
        n_val_images=args.n_val_images,
        n_test_images=args.n_test_images,
        task_weights_str=args.task_weights,
        loss_function_type=args.loss,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        num_conv_layers=args.num_conv_layers,
        head_dims_str=args.head_dims,
        root=args.root,
        graphs_dir=args.graphs_dir,
        y_labels_csv=args.y_labels_csv,
        split_json=args.split_json,
    )


