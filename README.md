# GCNN

# Packages
Installing required packages.

For running on a machine that has only cpu.
```
pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric
```

For serialization and storing the processed data for later sessions.
```
pip install pickle5
```

For proper formatting of code.
```
pip install git+git://github.com/psf/black
```

For visualizing structure connections.
```
pip install python-igraph
```

For Bayesian hyperparameter optimization.
'''
pip install ray[tune]
pip install hpbandster ConfigSpace
'''
# Code formatting
Before commiting and pushing changes, run command `black .` from the main directory of the project(GCNN).


# Running the code
The main file to run is main.py. When running for the first time uncomment the part with RawdatasetLoader on the beginning to process the raw files and create serialized objects.