#!/usr/bin/env python3

import json
import yaml

# TODO: create a pydantic.BaseModel to validate the config object
# TODO: add a version number to the config (and/or redo all configs...)
#
# TODO: refactor edge_features, task_weights, and Variables_of_interest
# TODO: change "output_heads" to be a list.


def group_features(tasks):
    """Given a task dictionary, list out
    all runs of a given type.  For example,
    two numeric tasks followed by three binary
    tasks would return
    names = ["name1 name2", "name3 name4 name5"]
    sizes = [2, 3]
    types = ["numeric", "binary"]
    """
    names = []
    sizes = []
    types = []

    name = ""
    sz = 0
    cur_type = None

    def finalize(v):
        # flush name,sz,cur_type to the list
        # and start a new tab
        nonlocal names, sizes, types, name, sz, cur_type
        if sz > 0:
            names.append(name)
            sizes.append(sz)
            types.append(cur_type)
        if v is None:
            return

        name = v["name"]
        sz = 1
        cur_type = v["type"]

    start = True
    for v in tasks:
        if start:
            finalize(v)
            start = False
        elif v["type"] != cur_type:
            finalize(v)
        else:
            name += " " + v["name"]
            sz += 1

    finalize(None)
    return names, sizes, types


def get_var_config(config, desc):
    """takes the pretrained config file and updates
    it according to the description file

    Args:
        config (dict): The pretrained model configuration to be updated.
        desc (str): A description or specification that dictates the changes
                    to be made to the configuration.

    Returns:
        dict: The updated configuration reflecting changes based on the description.
    """
    group_names, group_sizes, group_type = group_features(desc["graph_tasks"])
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    # var_config["node_feature_dims"] = config["NeuralNetwork"]["Variables_of_interest"]["input_node_feature_dims"]
    ninp = len(var_config["node_feature_dims"])
    var_config["input_node_features"] = list(range(ninp))

    ngrp = len(group_names)
    var_config["output_index"] = list(range(ngrp))
    var_config["type"] = ["graph"] * ngrp
    # list of regression / prediction targets
    var_config["output_names"] = [
        group_names[item] for ihead, item in enumerate(var_config["output_index"])
    ]
    # all graph features present in data
    var_config["graph_feature_names"] = group_names
    var_config["graph_feature_dims"] = group_sizes
    return var_config


def get_arc_config(config, desc):
    """takes pretrained config file and updates the architecture for fine-tuning
    based on desc. this is default and can be changed once the config is generated

    Args:
        config (dict): The pretrained model configuration.
        desc (str): A description or specification that guides how the architecture
                    should be updated for fine-tuning.

    Returns:
        dict: The updated configuration with fine-tuning adjustments.
    """
    group_names, group_sizes, group_type = group_features(desc["graph_tasks"])
    ntasks = sum(group_sizes)
    arc_config = {
        "output_heads": {
            "graph": {
                "dim_pretrained": 50,
                "num_headlayers": 2,
                "dim_headlayers": [50, 25],
            }
        },
        "task_weights": [1.0] * ntasks,
        "output_dim": [1] * ntasks,  # would only be > 1 if categorical, which is todo.
        "output_type": ["graph"],
    }
    return arc_config


def get_training_config():
    return {
        "num_epoch": 4,
        "EarlyStopping": True,
        "perc_train": 0.9,
        "loss_function_type": "mae",
        "batch_size": 32,
        "continue": 0,
        "Optimizer": {"type": "AdamW", "learning_rate": 1e-05},
        "conv_checkpointing": False,
    }


def main(argv):
    assert len(argv) == 4, f"Usage: {argv[0]} <in.yaml> <pre.json> <out.json>"
    inp = argv[1]
    pre = argv[2]
    out = argv[3]

    with open(inp, "r", encoding="utf-8") as f:
        descr = yaml.safe_load(f)

    with open(pre, "r") as f:
        config = json.load(f)
    # smiles: IsomericSMILES
    # split: split
    # graph_tasks:
    # - { name: alcoholic, type: binary, description: 'scent label' }
    # - { name: aldehydic, type: binary, description: 'scent label' }

    var_config = get_var_config(config, descr)
    arc_config = get_arc_config(config, descr)
    train_config = get_training_config()
    ft_config = {
        "NeuralNetwork": {"Architecture": config["NeuralNetwork"]["Architecture"]},
        "FTNeuralNetwork": {"Architecture": arc_config},
        "Variables_of_interest": var_config,
        "Training": train_config,
    }
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(ft_config, indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv)
