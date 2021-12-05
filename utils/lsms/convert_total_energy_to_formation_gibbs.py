##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import scipy.special
import math
import os
import shutil
import pandas
import numpy as np
import matplotlib.pyplot as plt

# function to return key for any value
def get_key(dictionary, val):

    for key, value in dictionary.items():
        if val == value:
            return key

    return None


def convert_raw_data_energy_to_gibbs(
    path_to_dir, elements_list, temperature_kelvin=0, create_plots=False
):
    # This works only for binary alloys

    new_dataset_path = path_to_dir[:-1] + "_gibbs_energy/"

    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)
    os.makedirs(new_dataset_path)

    Kb_joule_per_kelvin = 1.380649 * 1e-23
    conversion_joule_rydberg = 4.5874208973812 * 1e17
    Kb_rydberg_per_kelvin = Kb_joule_per_kelvin * conversion_joule_rydberg

    pure_elements_energy = dict()
    element_counter = dict()

    min_formation_enthalpy = float("Inf")
    max_formation_enthalpy = -float("Inf")

    total_energy_list = []
    linear_mixing_energy_list = []
    composition_list = []
    formation_enthalpy_list = []
    formation_gibbs_energy_list = []

    # Search for the configurations with pure elements and store their total energy
    for filename in os.listdir(path_to_dir):

        for atom_type in elements_list:
            element_counter[atom_type] = 0

        df = pandas.read_csv(path_to_dir + filename, header=None, nrows=1)
        energies = np.asarray([float(s) for s in df[0][0].split()])
        total_energy = energies[0]

        df = pandas.read_csv(path_to_dir + filename, header=None, skiprows=1)
        num_atoms = df.shape[0]
        for atom_index in range(0, num_atoms):
            row = df[0][atom_index].split()
            atom_type_str = int(row[0])
            element_counter[atom_type_str] += 1

        pure_element = get_key(element_counter, num_atoms)
        if pure_element is not None:
            pure_elements_energy[pure_element] = float(total_energy) / num_atoms

    # extract formation enthalpy from total energy
    # compute thermodynamic entropy
    # compute formation gibbs energy using formation enthalpy and thermodynamic entropy
    for filename in os.listdir(path_to_dir):

        (
            composition_element1,
            total_energy,
            linear_mixing_energy,
            formation_enthalpy,
        ) = compute_formation_enthalpy(
            path_to_dir + filename, elements_list, pure_elements_energy
        )

        # This is thermodynamic entropy, not statistical entropy
        # because we do not multiply the binomial coefficient by the probabilities
        entropy = Kb_rydberg_per_kelvin * math.log(
            scipy.special.comb(num_atoms, float(element_counter[elements_list[0]]))
        )

        formation_gibbs_energy = formation_enthalpy - temperature_kelvin * entropy

        min_formation_enthalpy = min(min_formation_enthalpy, formation_enthalpy)
        max_formation_enthalpy = max(max_formation_enthalpy, formation_enthalpy)

        total_energy_list.append(total_energy)
        linear_mixing_energy_list.append(linear_mixing_energy)
        composition_list.append(composition_element1)
        formation_enthalpy_list.append(formation_enthalpy)
        formation_gibbs_energy_list.append(formation_gibbs_energy)

        df = pandas.read_csv(path_to_dir + filename, header=None)
        df[0][0] = str(formation_gibbs_energy)
        df.to_csv(new_dataset_path + filename, header=None, index=None)

    print("Min formation enthalpy: ", min_formation_enthalpy)
    print("Max formation enthalpy: ", max_formation_enthalpy)

    if create_plots:
        plt.figure(0)
        plt.scatter(
            total_energy_list,
            linear_mixing_energy_list,
            edgecolor="b",
            facecolor="none",
        )
        plt.xlabel("Total energy (Rydberg)")
        plt.ylabel("Linear mixing energy (Rydberg)")
        plt.title("FePt")
        plt.savefig("parity.png")

        plt.figure(1)
        plt.scatter(
            composition_list, formation_enthalpy_list, edgecolor="b", facecolor="none"
        )
        plt.xlabel("Fe concentration")
        plt.ylabel("Formation enthalpy (Rydberg)")
        plt.title("FePt")
        plt.savefig("formation_enthalpy.png")

        plt.figure(2)
        plt.scatter(
            composition_list,
            formation_gibbs_energy_list,
            edgecolor="b",
            facecolor="none",
        )
        plt.xlabel("Fe concentration")
        plt.ylabel("Formation Gibbs energy (Rydberg)")
        plt.title("FePt")
        plt.savefig("formation_gibbs_energy.png")


def compute_formation_enthalpy(path_to_filename, elements_list, pure_elements_energy):

    # FIXME: this currently works only for binary alloys

    element_counter = dict()

    for atom_type in elements_list:
        element_counter[atom_type] = 0

    df = pandas.read_csv(path_to_filename, header=None, nrows=1)
    if type(df[0][0]) is str:
        energies = np.asarray([float(s) for s in df[0][0].split()])
        total_energy = energies[0]
    else:
        total_energy = df[0][0]

    df = pandas.read_csv(path_to_filename, header=None, skiprows=1)
    num_atoms = df.shape[0]

    for atom_index in range(0, num_atoms):
        row = df[0][atom_index].split()
        # If just an int() cast is used in the following line, a runtime error occurs
        # the following int(float()) cast is used to solve the runtime error
        # https://stackoverflow.com/questions/1841565/valueerror-invalid-literal-for-int-with-base-10
        atom_type_str = int(float(row[0]))
        element_counter[atom_type_str] += 1

    # count the occurrence of the first atom type
    element1 = elements_list[0]
    element2 = elements_list[1]
    composition_element1 = float(element_counter[element1]) / num_atoms

    # linear_minxing_energy = energy_elemet1 + (energy_element2 - energy_element1) * (1-element1)
    linear_mixing_energy = (
        pure_elements_energy[element1]
        + (pure_elements_energy[element2] - pure_elements_energy[element1])
        * (1 - composition_element1)
    ) * num_atoms

    formation_enthalpy = total_energy - linear_mixing_energy

    return composition_element1, total_energy, linear_mixing_energy, formation_enthalpy
