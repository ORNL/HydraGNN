This directory contains utilities for
wrangling datasets into useful
shape for HydraGNN to operate on them.


The first step is to describe your dataset
using a `descr.yaml` file.
The format of this file is given by example (FIXME).

Next, import your dataset into ADIOS2 format, using

    import_csv.py --input ../../../datasets/MolNet-ClinTox/clintox.csv --descr ../../../datasets/clintox.yaml --output data.bp

Then create a `config.json` file describing
the topology of your ML model.

    yaml_to_config.py ../../../datasets/clintox.yaml clintox.json

With both of these ingredients in-hand, you can run:

    train.py config.json data.bp

to produce trained model weights and predictions.

