from enum import Enum, auto


class AtomFeatures(Enum):
    NUM_OF_PROTONS = 0
    CHARGE_DENSITY = 1
    MAGNETIC_MOMENT = 2


class StructureFeatures(Enum):
    FREE_ENERGY = 0
    CHARGE_DENSITY = 1
    MAGNETIC_MOMENT = 2
