# HydraGNN

Distributed PyTorch implementation of multi-headed graph convolutional neural networks

<img src="https://github.com/ORNL/HydraGNN/assets/2488656/a6d5369b-2a70-4eee-aa39-b2cf8dedf262" alt="HydraGNN_QRcode" width="300" />


## Capabilities

- **Multi-headed prediction** for graph and node-level properties  
- **Distributed Data Parallelism**
- **Convolutional Layer** as a hyperparameter  
- **Equivariance** in convolution and prediction  
- **Global Attention** with linear scaling


## Dependencies

To install required packages with only basic capability (`torch`,
`torch_geometric`, and related packages)
and to serialize+store the processed data for later sessions (`pickle5`):
```
pip install -r requirements.txt
pip install -r requirements-torch.txt
pip install -r requirements-pyg.txt
```

If you plan to modify the code, include packages for formatting (`black`) and
testing (`pytest`) the code:
```
pip install -r requirements-dev.txt
```

Detailed dependency installation instructions are available on the
[Wiki](https://github.com/ORNL/HydraGNN/wiki/Install)


## Installation

After checking out HydgraGNN, we recommend to install HydraGNN in a
developer mode so that you can use the files in your current location
and update them if needed:
```
python -m pip install -e .
```

Or, simply type the following in the HydraGNN directory:
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

Alternatively, if you have no plane to update, you can install
HydraGNN in your python tree as a static package:
```
python setup.py install
```


## Running the code

There are two main options for running the code; both require a JSON input file
for configurable options.
1. Training a model, including continuing from a previously trained model using
configuration options:
```
import hydragnn
hydragnn.run_training("examples/configuration.json")
```
2. Making predictions from a previously trained model:
```
import hydragnn
hydragnn.run_prediction("examples/configuration.json", model)
```

### Datasets

Built in examples are provided for testing purposes only. One source of data to
create HydraGNN surrogate predictions is DFT output on the OLCF Constellation:
https://doi.ccs.ornl.gov/

Detailed instructions are available on the
[Wiki](https://github.com/ORNL/HydraGNN/wiki/Datasets)

### Configurable settings

HydraGNN uses a JSON configuration file (examples in `examples/`):

There are many options for HydraGNN; the dataset and model type are particularly
important:
 - `["Verbosity"]["level"]`: `0`, `1`, `2`, `3`, `4` (int)
 - `["Dataset"]["name"]`: `CuAu_32atoms`, `FePt_32atoms`, `FeSi_1024atoms` (str)

Additionally, many important arguments fall within the `["NeuralNetwork"]` section:

- `["NeuralNetwork"]`
  - `["Architecture"]`
    - `["mpnn_type"]`  
      Accepted types: `CGCNN`, `DimeNet`, `EGNN`, `GAT`, `GIN`, `MACE`, `MFC`, `PAINN`, `PNAEq`, `PNAPlus`, `PNA`, `SAGE`, `SchNet` (str)
    - `["num_conv_layers"]`  
      Examples: `1`, `2`, `3`, `4` ... (int)
    - `["hidden_dim"]`  
      Dimension of node embeddings during convolution (int)
    - `["output_heads"]`  
      Task types: `node`, `graph` (int)

  - `["Variables of Interest"]`
    - `["input_node_features"]`  
      Indices from nodal data used as inputs (int)
    - `["output_index"]`  
      Indices from data used as targets (int)
    - `["type"]`  
      Either `node` or `graph` (string)
    - `["output_dim"]`  
      Dimensions of prediction tasks (list)
    - *(include graphic)*

  - `["Training"]`
    - `["num_epoch"]`  
      Examples: `75`, `100`, `250` (int)
    - `["batch_size"]`  
      Examples: `16`, `32`, `64` (int)
    - `["Optimizer"]["learnin_rate"]`  
      Examples: `2e-3`, `0.005` (float)
    - `["compute_grad_energy"]`  
      Use the gradient of energy to predict forces (bool)


### Citations
"HydraGNN: Distributed PyTorch implementation of multi-headed graph convolutional neural networks", Copyright ID#: 81929619
https://doi.org/10.11578/dc.20211019.2

## Contributing

We encourage you to contribute to HydraGNN! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.
