from enum import Enum


class AtomFeatures(Enum):
    """Class is an enum that represents features of an atom. Values paired with names of features represent column
    indexes for each feature that are used in referencing them throughout the project.
    """

    NUM_OF_PROTONS = 0
    CHARGE_DENSITY = 1
    MAGNETIC_MOMENT = 2


class StructureFeatures(Enum):
    """Class is an enum that represents features of a structure. Values paired with names of features represent column
    indexes for each feature that are used in referencing them throughout the project.
    """

    FREE_ENERGY = 0
    CHARGE_DENSITY = 1
    MAGNETIC_MOMENT = 2


class Dataset(Enum):
    """Class is an enum that represents available datasets and their combinations."""

    FePt = "FePt"
    CuAu = "CuAu"
    CuAu_FePt_SHUFFLE = "CuAu_FePt_SHUFFLE"
    CuAu_TRAIN_FePt_TEST = "CuAu_TRAIN_FePt_TEST"
    FePt_TRAIN_CuAu_TEST = "FePt_TRAIN_CuAu_TEST"
    FeSi = "FeSi"
    FePt_FeSi_SHUFFLE = "CuAu_FePt_FeSi_SHUFFLE"
