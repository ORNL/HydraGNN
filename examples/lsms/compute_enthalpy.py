import os
import numpy as np
from scipy.interpolate import griddata

from tqdm import tqdm
import matplotlib.pyplot as plt

import shutil

total_energies_pure_elements = {26: 0.0, 78: 0.0}
list_pure_elements = [26, 78]
num_atoms = 32

total_energy_pure_elements = {}

def replace_first_line(source_dir, destination_dir, filename, new_line):
    # Construct the file paths for source and destination
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)

    # Read the original file
    with open(source_file, 'r') as f:
        lines = f.readlines()

    # Modify the first line
    if lines:
        lines[0] = new_line + '\n'

    # Write the modified content to the destination file
    with open(destination_file, 'w') as f:
        f.writelines(lines)


def compute_mixing_enthalpy(total_energy_pure_elements, chemical_composition, total_energy):

    concetration_Fe = chemical_composition.count(list_pure_elements[0]) / num_atoms
    concetration_Pt = 1-concetration_Fe

    mixing_enthalpy = total_energy - concetration_Fe * total_energy_pure_elements[list_pure_elements[0]] - concetration_Pt * total_energy_pure_elements[list_pure_elements[1]]

    return mixing_enthalpy



def read_LSMS_output(filepath):
    """Transforms lines of strings read from the raw data LSMS file to Data object and returns it.

    Parameters
    ----------
    lines:
      content of data file with all the graph information
    Returns
    ----------
    Data
        Data object representing structure of a graph sample.
    """

    f = open(filepath, "r", encoding="utf-8")

    lines = f.readlines()
    energies = lines[0].split(None, 2)

    # collect graph features
    total_energy = float(energies[0])

    chemical_composition = []
    atomic_magnetic_moment = []
    for line in lines[1:]:
        node_feat = line.split(None, 11)

        chemical_composition.append(int(node_feat[0]))
        atomic_magnetic_moment.append(float(node_feat[6]))

    f.close()

    total_magnetic_moment = sum(atomic_magnetic_moment)

    return total_energy, chemical_composition, total_magnetic_moment


def perform_histogram_cutoff(source_path, destination_path, histogram_cutoff):
    """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
     After that the serialized data is stored to the serialized_dataset directory.
     """

    assert (
            len(os.listdir(source_path)) > 0
    ), "No data files provided in {}!".format(source_path)

    chemical_compositions_list  = {}

    for i in range(num_atoms+1):
        chemical_compositions_list[i] = 0

    filelist = sorted(os.listdir(source_path))

    for name in tqdm(filelist):
        if name == ".DS_Store":
            continue
        # if the directory contains file, iterate over them
        if os.path.isfile(os.path.join(source_path, name)):
            _, chemical_composition, _ = read_LSMS_output(os.path.join(source_path, name))
            if chemical_compositions_list[chemical_composition.count(list_pure_elements[0])] < histogram_cutoff:
                shutil.copy2(os.path.join(source_path, name), destination_path)
                chemical_compositions_list[chemical_composition.count(list_pure_elements[0])] += 1
            else:
                continue


def generate_enthalpy_dataset(source_path, destination_path):
    """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
     After that the serialized data is stored to the serialized_dataset directory.
     """

    assert (
            len(os.listdir(source_path)) > 0
    ), "No data files provided in {}!".format(source_path)

    chemical_compositions_list  = {}

    total_energy_Pt, chemical_composition, _ = read_LSMS_output(os.path.join(source_path, 'out_0'))
    total_energy_Fe, chemical_composition, _ = read_LSMS_output(os.path.join(source_path, 'out_32017'))
    total_energy_pure_elements[list_pure_elements[0]] = total_energy_Fe
    total_energy_pure_elements[list_pure_elements[1]] = total_energy_Pt

    for i in range(num_atoms+1):
        chemical_compositions_list[i] = 0

    filelist = sorted(os.listdir(source_path))

    for name in tqdm(filelist):
        if name == ".DS_Store":
            continue
        # if the directory contains file, iterate over them
        if os.path.isfile(os.path.join(source_path, name)):
            total_energy, chemical_composition, _ = read_LSMS_output(os.path.join(source_path, name))
            mixing_enthalpy = compute_mixing_enthalpy(total_energy_pure_elements, chemical_composition, total_energy)
            replace_first_line(source_path, destination_path, name, str(mixing_enthalpy))


def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=xdata, y=ydata, bins=[nbin, nbin]
    )
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm



def plot_data(source_path):

    filelist = sorted(os.listdir(source_path))

    xdata = []
    mixing_enthalpy_list = []
    total_magnetic_moment_list = []

    for name in tqdm(filelist):

        if name == ".DS_Store":
            continue

        # if the directory contains file, iterate over them
        if os.path.isfile(os.path.join(source_path, name)):
            mixing_enthalpy, chemical_composition, total_magnetic_moment = read_LSMS_output(os.path.join(source_path, name))
            mixing_enthalpy_list.append(mixing_enthalpy)
            total_magnetic_moment_list.append(total_magnetic_moment)
            xdata.append(chemical_composition.count(list_pure_elements[0])/num_atoms)



    # plot mixing enthalpy as a function of chemical composition
    fig, ax = plt.subplots()
    hist2d_norm = getcolordensity(xdata, mixing_enthalpy_list)

    plt.scatter(
        xdata, mixing_enthalpy_list, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel('Fe concentration')
    plt.ylabel('Formation Energy (Ryd)')
    plt.title('FePt')
    ax.set_xticks([0.0, 0.5, 1.0])
    plt.draw()
    plt.tight_layout()
    plt.savefig("./BCT_enthalpy" + ".png", dpi=400)


    # plot total magnetic moment as a function of chemical composition
    fig, ax = plt.subplots()
    hist2d_norm = getcolordensity(xdata, total_magnetic_moment_list)

    plt.scatter(
        xdata, total_magnetic_moment_list, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel('Fe concentration')
    plt.ylabel('Total magnetic moment (magneton)')
    plt.title('FePt')
    ax.set_xticks([0.0, 0.5, 1.0])
    plt.draw()
    plt.tight_layout()
    plt.savefig("./magnetic_moment_vs_composition" + ".png", dpi=400)


    # plot total magnetic moment against mixing enthalpy
    fig, ax = plt.subplots()
    hist2d_norm = getcolordensity(total_magnetic_moment_list, mixing_enthalpy_list)

    plt.scatter(
        total_magnetic_moment_list, mixing_enthalpy_list, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel('Total magnetic moment (magneton)')
    plt.ylabel('Formation Energy (Ryd)')
    plt.title('FePt')
    ax.set_xticks([0.0, 0.5, 1.0])
    plt.draw()
    plt.tight_layout()
    plt.savefig("./mixing_enthalpy_vs_magnetic_moment" + ".png", dpi=400)


if __name__ == "__main__":
    raw_data_path = "./output_files"
    destination_path1 = "./FePt"
    destination_path2 = "./FePt_enthalpy"
    histogram_cutoff = 1000

    if os.path.exists(destination_path1):
        shutil.rmtree(destination_path1)
    os.makedirs(destination_path1)

    if os.path.exists(destination_path2):
        shutil.rmtree(destination_path2)
    os.makedirs(destination_path2)

    perform_histogram_cutoff(raw_data_path, destination_path1, histogram_cutoff)
    generate_enthalpy_dataset(destination_path1, destination_path2)
    plot_data(destination_path2)
