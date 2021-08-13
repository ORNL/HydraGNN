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

    FePt = "FePt_32atoms"
    CuAu = "CuAu_32atoms"
    FeSi = "FeSi_1024atoms"
    unit_test = "unit_test"
