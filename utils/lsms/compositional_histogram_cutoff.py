import scipy.special
import math
import os
import shutil
import pandas
import numpy as np
import matplotlib.pyplot as plt


def compositional_cutoff_histogram(path_to_dir, elements_list, histogram_cutoff=1000):

    new_dataset_path = path_to_dir[:-1] + "_histogram_cutoff/"

    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)
    os.makedirs(new_dataset_path)

    element_counter = dict()
    composition_counter = dict()

    for filename in os.listdir(path_to_dir):

        for atom_type in elements_list:
            element_counter[atom_type] = 0

        df = pandas.read_csv(path_to_dir + filename, header=None, skiprows=1)
        num_atoms = df.shape[0]
        for atom_index in range(0, num_atoms):
            row = df[0][atom_index].split()
            atom_type_str = row[0]
            element_counter[atom_type_str] += 1

        if element_counter[elements_list[0]] in composition_counter:
            composition_counter[element_counter[elements_list[0]]] = (
                composition_counter[element_counter[elements_list[0]]] + 1
            )
        else:
            composition_counter[element_counter[elements_list[0]]] = 1

        if composition_counter[element_counter[elements_list[0]]] <= histogram_cutoff:
            df = pandas.read_csv(path_to_dir + filename, header=None)
            df.to_csv(new_dataset_path + filename, header=None, index=None)


if __name__ == "__main__":
    compositional_cutoff_histogram(
        "./FePt/", elements_list=["26", "78"], histogram_cutoff=1000
    )
