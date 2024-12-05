import torch
from torch_geometric.data import Data
from hydragnn.utils.print.print_utils import print_distributed, iterate_tqdm
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_sampling(dataset: [Data], subsample_percentage: float, verbosity=0):
    """Given the datasets and the percentage of data you want to extract from it, method will
    apply stratified sampling where X is the datasets and Y is are the category values for each datapoint.
    In the case of the structures datasets where each structure contains 2 types of atoms, the category will
    be constructed in a way: number of atoms of type 1 + number of protons of type 2 * 100.

    Parameters
    ----------
    dataset: [Data]
        A list of Data objects representing a structure that has atoms.
    subsample_percentage: float
        Percentage of the datasets.

    Returns
    ----------
    [Data]
        Subsample of the original datasets constructed using stratified sampling.
    """
    dataset_categories = []
    print_distributed(verbosity, "Computing the categories for the whole datasets.")
    for data in iterate_tqdm(dataset, verbosity):
        frequencies = torch.bincount(data.x[:, 0].int())
        frequencies = sorted(frequencies[frequencies > 0].tolist())
        category = 0
        for index, frequency in enumerate(frequencies):
            category += frequency * (100 ** index)
        dataset_categories.append(category)

    subsample_indices = []
    subsample = []

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=subsample_percentage, random_state=0
    )

    for subsample_index, _ in sss.split(dataset, dataset_categories):
        subsample_indices = subsample_index.tolist()

    for index in subsample_indices:
        subsample.append(dataset[index])

    return subsample
