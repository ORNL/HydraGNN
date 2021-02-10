# GCNN

# Packages
Installing required packages.

For running on a machine that has only cpu.
```
pip3 install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-geometric
```

For serialization and storing the processed data for later sessions.
```
pip3 install pickle5
```

For proper formatting of code.
```
pip3 install git+git://github.com/psf/black
```

For tracking down progress using progress bar.
```
pip3 install tqdm
```
    
For visualizing structure connections.
```
pip3 install python-igraph
```

For Bayesian hyperparameter optimization.
```
pip3 install ray[tune]
pip3 install hpbandster ConfigSpace
pip3 install -U hyperopt
```
# Code formatting
Before commiting and pushing changes, run command `black .` from the main directory of the project(GCNN).


# Running the code
There are two options for running the code:
    1. Training the model(main.py)
    2. Reproducing the results using existing models(test_trained_models.py)


## Training the model(main.py)
There are 3 options:
    1. Run it with hyperparameter optimization
    2. Run it normally with inputting the parameters of data and model from terminal
    3. Run it normally but load the whole parameters config from json file whose location is GCNN/utilities/configuration.json

### Loading parameters from configuration.json file
This is an example for configuration.json file:
```
{
    "atom_features": [
        0,
        1,
        2
    ],
    "batch_size": 64,
    "hidden_dim": 15,
    "num_conv_layers": 16,
    "learning_rate": 0.02,
    "radius": 10,
    "max_num_node_neighbours": 10,
    "num_epoch": 200,
    "perc_train": 0.7,
    "predicted_value_option": 1,
    "dataset_option": "FePt",
    "model_type": "PNN"
}
```
Possible values:
    1. atom_features(0, 1, 2) where 0 = proton number, 1 = charge density, 2 = magnetic moment
    2. batch size(int, >0) usually power of 2
    3. hidden_dim(int, >0) size of a hidden layer
    4. num_conv_layers(int, >0) number of convolutional layers
    5. radius(int, >5) radius from which neighbours of an node are chosen, represents distance in 3D space
    6. max_num_node_neighbours(int, >0) maximum number of node neighbours
    7. num_epoch(int, >0) number of epochs to train the model
    8. perc_train(float, >0, <1) percentage of dataset used for training, the rest(1-perc_train) is split equally for validation and testing
    9. predicted_value_option(1, 2, 3, 4, 5, 6) where 1 = free energy, 2 = charge_density, 3 = magnetic moment, 4 = free energy+charge density 5 = free energy+magnetic moment, 6 = free energy+charge density+magnetic moment
    10. dataset_option(str) possible values: FePt, CuAu, FeSi, CuAu_FePt_SHUFFLE, CuAu_TRAIN_FePt_TEST, FePt_TRAIN_CuAu_TEST
    11. model_type(str) possible values PNN, MFC, GIN and GAT
    12. subsample_size(int) needs to be smaller than the size of original dataset, when used the serialized dataset loader will take the subset of the original dataset using stratified sampling

Remark: keep in mind that predicted_value_option is used to compute output dimension. Besides free energy which has dimension of 1, other combinations depend on the number of atoms in the dataset(example: free energy+charge density=1 + number of atoms) and this was configured specifically for structures that have number of atoms equal to 32. If you want to add datasets with different structures in terms of number of atoms, you need to change parts of the code in main.py where this is computed from a dictionary.
