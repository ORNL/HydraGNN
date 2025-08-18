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

LOCAL_DATA_DIR = './tmp'
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
os.system(f'mkdir -p {LOCAL_DATA_DIR}/qcml/dft_force_field/')
os.system(
    f'gcloud storage cp -r {QCML_DATA_DIR}/qcml/dft_force_field/1.0.0'
    f' {LOCAL_DATA_DIR}/qcml/dft_force_field/ --project={GCP_PROJECT}'
)
force_field_ds = tfds.load(
    'qcml/dft_force_field', split='full', data_dir=LOCAL_DATA_DIR
)
force_field_iter = iter(force_field_ds)
example = next(force_field_iter)
print(example)

# ===================================================================
# Example 2: Combine 'dft_force_field' with 'dft_d4_correction'
# ===================================================================
os.system(f'mkdir -p {LOCAL_DATA_DIR}/qcml/dft_d4_correction/')
os.system(
    f'gcloud storage cp -r {QCML_DATA_DIR}/qcml/dft_d4_correction/1.0.0'
    f' {LOCAL_DATA_DIR}/qcml/dft_d4_correction/ --project={GCP_PROJECT}'
)
# Note the read config to keep the same record order in both datasets.
read_config = tfds.ReadConfig(interleave_cycle_length=1)
force_field_ds_for_zip = tfds.load(
    'qcml/dft_force_field',
    split='full',
    data_dir=LOCAL_DATA_DIR,
    read_config=read_config,
)
d4_correction_ds_for_zip = tfds.load(
    'qcml/dft_d4_correction',
    split='full',
    data_dir=LOCAL_DATA_DIR,
    read_config=read_config,
)
zipped_ds = tf.data.Dataset.zip(
    force_field_ds_for_zip, d4_correction_ds_for_zip
)
zipped_iter = iter(zipped_ds)

# The example contains one tuple element (feature dict) per input dataset.
example = next(zipped_iter)
print('atomic_numbers from first', example[0]['atomic_numbers'])
print('d4_energy from second', example[1]['d4_energy'])
# The feature 'key_hash' can be used to verify the correct example order.
print('Matching key_hashes', [t['key_hash'] for t in example])