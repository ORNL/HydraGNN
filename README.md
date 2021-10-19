# HydraGNN

Distributed PyTorch implementation of multi-headed graph convolutional neural networks

## Dependencies

To install required packages with only basic capability (`torch`,
`torch_geometric`, and related packages)
and to serialize+store the processed data for later sessions (`pickle5`):
```
pip install -r requirements.txt
pip install -r requirements-torchdep.txt
```

If you plan to modify the code, include packages for formatting (`black`) and
testing (`pytest`) the code:
```
pip install -r requirements-dev.txt
```

Detailed dependency installation instructions are available on the
[Wiki](https://github.com/ORNL/HydraGNN/wiki/Install)

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
 - `["Verbosity"]["level"]`: `0`, `1`, `2`, `3`, `4`
 - `["Dataset"]["name"]`: `CuAu_32atoms`, `FePt_32atoms`, `FeSi_1024atoms`
 - `["NeuralNetwork"]["Architecture"]["model_type"]`: `PNA`, `MFC`, `GIN`, `GAT`, `CGCNN`

## Contributing

We encourage you to contribute to HydraGNN! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.
