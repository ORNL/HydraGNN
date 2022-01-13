import os
import shutil
import numpy as np
from tqdm import tqdm
from sympy.utilities.iterables import multiset_permutations
import scipy.special
import math


def write_to_file(total_energy, atomic_features, count_config, dir):

    numpy_string_total_value = np.array2string(total_energy)

    filetxt = numpy_string_total_value

    for index in range(0, atomic_features.shape[0]):
        numpy_row = atomic_features[index, :]
        numpy_string_row = np.array2string(
            numpy_row, precision=2, separator="\t", suppress_small=True
        )
        filetxt += "\n" + numpy_string_row.lstrip("[").rstrip("]")

        filename = os.path.join(dir, "output" + str(count_config) + ".txt")
        with open(filename, "w") as f:
            f.write(filetxt)


# 3D Ising model
def E_dimensionless(config, L):
    total_energy = 0

    count_pos = 0
    number_nodes = L ** 3
    positions = np.zeros((number_nodes, 3))
    atomic_features = np.zeros((number_nodes, 5))
    for z in range(L):
        for y in range(L):
            for x in range(L):
                positions[count_pos, 0] = x
                positions[count_pos, 1] = y
                positions[count_pos, 2] = z

                S = config[x, y, z]
                nb = (
                    config[(x + 1) % L, y, z]
                    + config[x, (y + 1) % L, z]
                    + config[(x - 1) % L, y, z]
                    + config[x, (y - 1) % L, z]
                    + config[x, y, z]
                    + config[x, y, (z + 1) % L]
                    + config[x, y, (z - 1) % L]
                )
                total_energy += -nb * S

                atomic_features[count_pos, 0] = config[x, y, z]
                atomic_features[count_pos, 1:4] = positions[count_pos, :]
                atomic_features[count_pos, 4] = math.sin(math.pi * config[x, y, z] / 2)

                count_pos = count_pos + 1

    total_energy = total_energy / 6

    return total_energy, atomic_features


def create_dataset(L, histogram_cutoff, dir):

    count_config = 0

    for num_downs in tqdm(range(0, L ** 3)):

        primal_configuration = np.ones((L ** 3,))
        for down in range(0, num_downs):
            primal_configuration[down] = -1.0

        # If the current composition has a total number of possible configurations above
        # the hard cutoff threshold, a random configurational subset is picked
        if scipy.special.binom(L ** 3, num_downs) > histogram_cutoff:
            for num_config in range(0, histogram_cutoff):
                random_scaling = np.random.random((L ** 3,))
                config = np.random.permutation(primal_configuration)
                config = np.multiply(config, random_scaling)
                config = np.reshape(config, (L, L, L))
                total_energy, atomic_features = E_dimensionless(config, L)

                write_to_file(total_energy, atomic_features, count_config, dir)

                count_config = count_config + 1

        # If the current composition has a total number of possible configurations smaller
        # than the hard cutoff, then all possible permutations are generated
        else:
            for config in multiset_permutations(primal_configuration):
                config = np.array(config)
                config = np.reshape(config, (L, L, L))
                total_energy, atomic_features = E_dimensionless(config, L)

                write_to_file(total_energy, atomic_features, count_config, dir)

                count_config = count_config + 1


if __name__ == "__main__":

    dir = os.path.join(os.path.dirname(__file__), "../../dataset/ising_model")
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    number_atoms_per_dimension = 3
    configurational_histogram_cutoff = 1000

    create_dataset(number_atoms_per_dimension, configurational_histogram_cutoff, dir)
