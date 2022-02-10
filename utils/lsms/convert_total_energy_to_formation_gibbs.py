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
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# This is LSMS specific - it assumes only one header line and only atoms following.
def read_file(path):
    with open(path, "r") as rf:
        txt = rf.readlines()
    total_energy_txt = txt[0].split()[0]

    return total_energy_txt, txt


def convert_raw_data_energy_to_gibbs(
    dir, elements_list, temperature_kelvin=0, overwrite_data=False, create_plots=True
):
    # NOTE: This works only for binary alloys

    if dir.endswith("/"):
        dir = dir[:-1]
    new_dir = dir + "_gibbs_energy/"

    if os.path.exists(new_dir):
        if overwrite_data:
            shutil.rmtree(new_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    elements_list = sorted(elements_list)
    pure_elements_energy = dict()

    min_formation_enthalpy = float("Inf")
    max_formation_enthalpy = -float("Inf")

    # Search for the configurations with pure elements and store their total energy
    all_files = os.listdir(dir)
    for filename in tqdm(all_files):

        path = os.path.join(dir, filename)
        total_energy, txt = read_file(path)
        atoms = np.loadtxt(txt[1:])
        pure_element = np.unique(atoms[:, 0])
        if len(pure_element) == 1:
            num_atoms = atoms.shape[0]
            pure_elements_energy[pure_element[0]] = float(total_energy) / num_atoms

    assert len(pure_elements_energy) == 2, "Must have two single element files."

    num_files = len(all_files)
    total_energy_list = np.ndarray(num_files)
    linear_mixing_energy_list = np.ndarray(num_files)
    composition_list = np.ndarray(num_files)
    formation_enthalpy_list = np.ndarray(num_files)
    formation_gibbs_energy_list = np.ndarray(num_files)

    # extract formation enthalpy from total energy
    # compute thermodynamic entropy
    # compute formation gibbs energy using formation enthalpy and thermodynamic entropy
    for fn, filename in enumerate(tqdm(all_files)):

        path = os.path.join(dir, filename)
        total_energy_txt, txt = read_file(path)
        atoms = np.loadtxt(txt[1:])

        (
            composition_element1,
            total_energy,
            linear_mixing_energy,
            formation_enthalpy,
            entropy,
        ) = compute_formation_enthalpy(
            path, elements_list, pure_elements_energy, float(total_energy_txt), atoms
        )

        formation_gibbs_energy = formation_enthalpy - temperature_kelvin * entropy

        if create_plots:
            total_energy_list[fn] = total_energy
            linear_mixing_energy_list[fn] = linear_mixing_energy
            composition_list[fn] = composition_element1
            formation_enthalpy_list[fn] = formation_enthalpy
        formation_gibbs_energy_list[fn] = formation_gibbs_energy

        # Replace the total energy with mixing.
        txt[0] = txt[0].replace(total_energy_txt, str(formation_gibbs_energy))
        new_path = os.path.join(new_dir, filename)
        with open(new_path, "w") as wf:
            wf.write("".join(txt))

    min_formation_enthalpy = np.min(formation_gibbs_energy_list)
    max_formation_enthalpy = np.max(formation_gibbs_energy_list)
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
        plt.savefig("linear_mixing_energy.png")

        plt.figure(1)
        plt.scatter(
            composition_list, formation_enthalpy_list, edgecolor="b", facecolor="none"
        )
        plt.xlabel("Concentration")
        plt.ylabel("Formation enthalpy (Rydberg)")
        plt.savefig("formation_enthalpy.png")

        plt.figure(2)
        plt.scatter(
            composition_list,
            formation_gibbs_energy_list,
            edgecolor="b",
            facecolor="none",
        )
        plt.xlabel("Concentration")
        plt.ylabel("Formation Gibbs energy (Rydberg)")
        plt.savefig("formation_gibbs_energy.png")


def compute_formation_enthalpy(
    path, elements_list, pure_elements_energy, total_energy, atoms
):

    # FIXME: this currently works only for binary alloys

    elements, counts = np.unique(atoms[:, 0], return_counts=True)

    # Check all systems are in this binary.
    for e in elements:
        assert (
            e in elements_list
        ), "Sample {} contains element not present in binary considered.".format(path)
    # Fixup for the pure component cases.
    for e, elem in enumerate(elements_list):
        if elem not in elements:
            elements = np.insert(elements, e, elem)
            counts = np.insert(counts, e, 0)

    # Use first element for composition.
    num_atoms = atoms.shape[0]
    composition = counts[0] / num_atoms

    # linear_mixing_energy = energy_element1 + (energy_element2 - energy_element1) * (1-element1)
    linear_mixing_energy = (
        pure_elements_energy[elements[0]] * composition
        + pure_elements_energy[elements[1]] * (1 - composition)
    ) * num_atoms

    formation_enthalpy = total_energy - linear_mixing_energy

    # LSMS units are fixed.
    Kb_joule_per_kelvin = 1.380649 * 1e-23
    conversion_joule_rydberg = 4.5874208973812 * 1e17
    Kb_rydberg_per_kelvin = Kb_joule_per_kelvin * conversion_joule_rydberg

    # This is thermodynamic entropy, not statistical entropy
    # because we do not multiply the binomial coefficient by the probabilities
    entropy = Kb_rydberg_per_kelvin * math.log(scipy.special.comb(num_atoms, counts[0]))

    return composition, total_energy, linear_mixing_energy, formation_enthalpy, entropy
