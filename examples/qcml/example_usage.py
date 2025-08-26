# Copyright 2024 DeepMind Technologies Limited

# All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
# you may not use this file except in compliance with the Apache 2.0 license.
# You may obtain a copy of the Apache 2.0 license at:
# https://www.apache.org/licenses/LICENSE-2.0

# All other materials are licensed under the Creative Commons Attribution 4.0
# International License (CC BY-NC). You may obtain a copy of the CC BY-NC
# license at: https://creativecommons.org/licenses/by-nc/4.0/legalcode

# Unless required by applicable law or agreed to in writing, all software and
# materials distributed here under the Apache 2.0 or CC BY-NC licenses are
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the licenses for the specific language
# governing permissions and limitations under those licenses.

# This is not an official Google product.

import os
import tensorflow as tf
import tensorflow_datasets as tfds

from sys import argv
try:
    from mpi4py import MPI
except ImportError:
    raise SystemExit("Please `pip install mpi4py`")

def balanced_block(rank: int, world_size: int, N: int):
    """
    Return [start, end) for this rank: consecutive indices start..end-1.
    Handles any N and world_size, including N < world_size.
    """
    base = N // world_size           # minimum chunk size
    extra = N % world_size           # number of ranks that get one extra item

    if rank < extra:
        start = rank * (base + 1)
        end   = start + (base + 1)
    else:
        start = extra * (base + 1) + (rank - extra) * base
        end   = start + base

    return start, end  # half-open interval

LOCAL_DATA_DIR = './dataset'
QCML_DATA_DIR = 'gs://qcml-datasets/tfds'
GCP_PROJECT = 'deepmind-opensource'

# Installation of the used 'gcloud': https://cloud.google.com/sdk/docs/install

# ===========================
# No authentication necessary
# ===========================
# Alternatively, see https://cloud.google.com/docs/authentication/gcloud.
os.system('gcloud config set auth/disable_credentials True')

# =============================================
# Example 1: Feature group 'dft_force_field'
# =============================================
# TFDS directory structure: <data_dir>/<dataset>/<builder_config>/<version>.
#os.system(f'mkdir -p {LOCAL_DATA_DIR}/qcml/dft_force_field/')
#os.system(
#    f'gcloud storage cp -r {QCML_DATA_DIR}/qcml/dft_force_field/1.0.0'
#    f' {LOCAL_DATA_DIR}/qcml/dft_force_field/ --project={GCP_PROJECT}'
#)
force_field_ds = tfds.load(
    'qcml/dft_force_field', split='full', data_dir=LOCAL_DATA_DIR
)
force_field_iter = iter(force_field_ds)
example = next(force_field_iter)
print(example)
print(force_field_ds.cardinality())


# Enumerate the dataset to add an index (ID) to each element
# The elements become tuples of (id, element)
force_field_ds_with_ids = force_field_ds.enumerate()


#for i, element in force_field_ds_with_ids:
#    print(f"ID: {i.numpy()}, Element: {element}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = force_field_ds.cardinality()
start, end = balanced_block(rank, size, N)
print(f"rank {rank}/{size-1}: indices [{start}, {end})  (count={end-start})")

list_samples_ids = [int(i) for i in range(start,end)]

# Convert the list of IDs into a lookup table for efficient filtering
keys_tensor = tf.constant(list_samples_ids, dtype=tf.int64)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, keys_tensor),
    default_value=-1
)

# Define the filter function
def filter_by_id(index, element):
    # Lookup the index in the table; it will return -1 if not found
    is_in_table = table.lookup(index)
    # The sample is a match if its index is not -1
    return is_in_table != -1


# Filter the dataset using the lookup table
filtered_dataset = force_field_ds_with_ids.filter(filter_by_id)

# Iterate through the filtered dataset and collect the results
for index, element in filtered_dataset:
    print(f"ID: {index.numpy()}, Element: {element}")
    print(element['positions'])

