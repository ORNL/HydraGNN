# GCNN

## Dependencies

To install required packages with only basic capability (`torch`,
`torch_geometric`, and related packages):
```
pip install -r requirements.txt
pip install -r requirements-torchdep.txt
```

In order to run Bayesian hyperparameter optimization additionally install:
```
pip install -r requirements-hyperopt.txt
```

If you plan to modify the code, include packages for formatting (`black`) and
testing (`pytest`) the code:
```
pip install -r requirements-dev.txt
```

For serialization and storing the processed data for later sessions (`pickle`)
and visualizing structure connections (`igraph`):
```
pip install -r requirements-other.txt
```

Detailed dependency installation instructions are available on the
[Wiki](https://github.com/allaffa/GCNN/wiki/Install)

## Running the code

There are two main options for running the code:
1. Training a model (including starting from a previously trained model using
configuration options)
    ```
    python run_config_file.py
    ```
2. Hyperparameter optimization
    ```
    python run_hyperparam_opt.py
    ```

### Datasets

Built in examples are provided for testing purposes only. One source of data to
create GCNN surrogate predictions is DFT output on the OLCF Constellation:
https://doi.ccs.ornl.gov/

Detailed instructions are available on the
[Wiki](https://github.com/allaffa/GCNN/wiki/Datasets)

### Configurable settings

GCNN uses a JSON configuration file (examples in `examples/`):

There are many options for GCNN; the dataset and model type are particularly
important:
 - `["Dataset"]["name"]`: `CuAu_32atoms`, `FePt_32atoms`, `FeSi_1024atoms`
 - `["NeuralNetwork"]["Architecture"]["model_type"]`: `PNA`, `MFC`, `GIN`, `GAT`

## Contributing

We encourage you to contribute to GCNN! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.
