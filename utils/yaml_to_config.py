#!/usr/bin/env python3

import json
import yaml

# TODO: create a pydantic.BaseModel to validate the config object
# TODO: add a version number to the config (and/or redo all configs...)
#
# TODO: refactor edge_features, task_weights, and Variables_of_interest
# TODO: change "output_heads" to be a list.
config = {
  "Verbosity": {
    "level": 2
  },
  "NeuralNetwork": {
    "Architecture": {
      "edge_features": ["aromatic", "single", "double", "triple"],
      "model_type": "PNA",
      "max_neighbours": 20,
      "hidden_dim": 200,
      "num_conv_layers": 4,
      "output_heads": {
        "graph": {
          "num_sharedlayers": 1,
          "dim_sharedlayers": 200,
          "num_headlayers": 2,
          "dim_headlayers": [
            200,
            200
          ]
        }
      },
      "task_weights": [
        1.0
      ]
    },
    "Variables_of_interest": {
      "denormalize_output": False
    },
    "Training": {
      "num_epoch": 10,
      "batch_size": 128,
      "continue": 0,
      "startfrom": "existing_model",
      "Optimizer": {
        "learning_rate": 0.001
      }
    }
  }
}

def group_features(tasks):
    """ Given a task dictionary, list out
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

def main(argv):
    assert len(argv) == 3, f"Usage: {argv[0]} <in.yaml> <out.json>"
    inp = argv[1]
    out = argv[2]

    with open(inp, "r", encoding="utf-8") as f:
        descr = yaml.safe_load(f)
    # smiles: IsomericSMILES
    # split: split
    # graph_tasks:
    # - { name: alcoholic, type: binary, description: 'scent label' }
    # - { name: aldehydic, type: binary, description: 'scent label' }

    group_names, group_sizes, group_type = group_features(descr["graph_tasks"])
    for i in (group_names, group_sizes, group_type):
        print(i)

    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    var_config["node_feature_dims"] = var_config["input_node_feature_dims"]
    ninp = len(var_config["node_feature_dims"])
    var_config["input_node_features"] = list(range(ninp))

    ngrp = len(group_names)
    var_config["output_index"] = list(range(ngrp))
    var_config["type"] = [ "graph" ]*ngrp
    # list of regression / prediction targets
    var_config["output_names"] = [
        group_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    # all graph features present in data
    var_config["graph_feature_names"] = group_names
    var_config["graph_feature_dims"]  = group_sizes

    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2))

if __name__=="__main__":
    import sys
    main(sys.argv)
