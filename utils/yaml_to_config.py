#!/usr/bin/env python3

import json
import yaml

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
        1
      ]
    },
    "Variables_of_interest": {
      "input_node_feature_names": ["Z", "x"],
      "input_node_feature_dims": [1, 3],
      "input_node_features": [0, 1],
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

def main(argv):
    assert len(argv) == 3, f"Usage: {argv[0]} <in.yaml> <out.json>"
    inp = argv[1]
    out = argv[2]

    graph_feature_names = ["CT_TOX"]
    graph_feature_dim = [1]

    with open(inp, "r", encoding="utf-8") as f:
        descr = yaml.safe_load(f)

    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["output_index"] = [ 0 ]
    var_config["type"] = [ "graph" ]
    var_config["output_names"] = [
        graph_feature_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dim

    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2))

if __name__=="__main__":
    import sys
    main(sys.argv)
