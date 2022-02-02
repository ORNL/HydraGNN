import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def find_bin(comp, nbins):
    bins = np.linspace(0, 1, nbins)
    for bi in range(len(bins) - 1):
        if comp > bins[bi] and comp < bins[bi + 1]:
            return bi
    return nbins - 1


def compositional_histogram_cutoff(
    dir,
    elements_list,
    histogram_cutoff,
    num_bins,
    overwrite_data=False,
    create_plots=True,
):

    if dir.endswith("/"):
        dir = dir[:-1]
    new_dir = dir + "_histogram_cutoff/"

    if os.path.exists(new_dir):
        if overwrite_data:
            shutil.rmtree(new_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    comp_final = []
    comp_all = np.zeros([num_bins])
    for filename in tqdm(os.listdir(dir)):

        path = os.path.join(dir, filename)
        # This is LSMS specific - it assumes only one header line and only atoms following.
        atoms = np.loadtxt(path, skiprows=1)

        elements, counts = np.unique(atoms[:, 0], return_counts=True)
        num_atoms = atoms.shape[0]
        # This will result in composition of 1.0 for any pure element.
        composition = counts[0] / num_atoms

        b = find_bin(composition, num_bins)
        comp_all[b] += 1
        if comp_all[b] < histogram_cutoff:
            comp_final.append(composition)
            new_path = os.path.join(new_dir, filename)
            os.symlink(path, new_path)

    if create_plots:
        plt.figure(0)
        plt.hist(comp_final, bins=num_bins)
        plt.savefig("composition_histogram_cutoff.png")

        plt.figure(1)
        w = 1 / num_bins
        plt.bar(np.linspace(0, 1, num_bins), comp_all, width=w)
        plt.savefig("composition_initial.png")
