import warnings
import numpy as np
import torch
from mendeleev import element as mendeleev_element
from tqdm import tqdm


class ChemicalFeatureEncoder:
    """Encodes chemical features for atomic elements using mendeleev properties."""

    def __init__(self, num_elements: int = 118):  # 118 elements in periodic table
        warnings.filterwarnings("ignore")
        self.block_map = {"s": 0, "p": 1, "d": 2, "f": 3}
        print("Generating element properties for chemical encoder ...")
        self.mel_props = {}
        for z in tqdm(range(1, num_elements + 1)):
            self.mel_props[z] = mendeleev_element(z)

    @staticmethod
    def normalize_features(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + eps)

    def compute_chem_features(self, data, atomic_num_idx: int = 0):
        x = data.x.cpu().numpy() if torch.is_tensor(data.x) else data.x
        Zs = x[:, atomic_num_idx]
        features = []
        for z in Zs:
            mel = self.mel_props.get(int(z))
            atomic_weight = float(mel.atomic_weight or 0.0)
            group_id = float(mel.group_id or 0)
            period = float(mel.period or 0)
            block_str = getattr(mel, "block", "").lower()
            block_id = float(self.block_map.get(block_str, -1))
            nvalence = float(mel.nvalence() or 0)
            cov_rad = float(mel.covalent_radius or 0.0)
            vdw_rad = float(mel.vdw_radius or 0.0)
            en_pauling = float(mel.en_pauling or 0.0)
            en_allen = float(mel.en_allen or 0.0)
            electron_affinity = float(mel.electron_affinity or 0.0)
            first_ionization_energy = float(mel.ionenergies.get(1, 0.0))
            melting_point = float(mel.melting_point or 0.0)
            boiling_point = float(mel.boiling_point or 0.0)
            density = float(mel.density or 0.0)
            atomic_volume = float(mel.atomic_volume or 0.0)
            mendeleev_feats = [
                atomic_weight,
                group_id,
                period,
                block_id,
                nvalence,
                cov_rad,
                vdw_rad,
                en_pauling,
                en_allen,
                electron_affinity,
                first_ionization_energy,
                melting_point,
                boiling_point,
                density,
                atomic_volume,
            ]
            features.append(mendeleev_feats)

        features = self.normalize_features(np.array(features, dtype=np.float32))
        data.ce = torch.tensor(features, dtype=torch.float32)
        return data
