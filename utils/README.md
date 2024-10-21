This directory contains utilities for
wrangling datasets into useful
shape for HydraGNN to operate on them.


The first step is to describe your dataset
using a `descr.yaml` file.
The format of this file is given by example (FIXME).

Next, import your dataset into ADIOS2 format, using

    import_csv.py --input ../../../datasets/MolNet-ClinTox/clintox.csv --descr ../../../datasets/clintox.yaml --output data.bp

Then create a `finetune_config.json` file describing
the topology of your fine-tuning heads based
 on the tasks defined in the model.  It requires
pointing to a pretrained model. 

    yaml_to_config.py <path_to_data_description>.yaml <pretrained_config>.json <data_finetuning_config>.json

The `finetune_config.json` file can subsequently be modified as desired. 
With both of these ingredients in-hand, you can run:

    train.py finetune_config.json data.bp

to produce trained model weights and predictions.

