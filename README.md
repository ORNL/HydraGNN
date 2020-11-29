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

For visualizing structure connections.
```
pip3 install python-igraph
```

For Bayesian hyperparameter optimization.
'''
pip3 install ray[tune]
pip3 install hpbandster ConfigSpace
pip3 install -U hyperopt
'''
# Code formatting
Before commiting and pushing changes, run command `black .` from the main directory of the project(GCNN).


# Running the code
The main file to run is main.py. When running for the first time uncomment the part with RawdatasetLoader on the beginning to process the raw files and create serialized objects.