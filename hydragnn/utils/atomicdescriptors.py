import os
import torch
import torch.nn.functional as F
import json

try:
    import mendeleev
except ImportError:
    pass


class atomicdescriptors:
    def __init__(
        self,
        embeddingfilename,
        overwritten=True,
        element_types=["C", "H", "O", "N", "F", "S"],
    ):
        if os.path.exists(embeddingfilename) and not overwritten:
            print("loading from existing file: ", embeddingfilename)
            with open(embeddingfilename, "r") as f:
                self.atom_embeddings = json.load(f)
        else:
            self.atom_embeddings = {}
            if element_types is None:
                self.element_types = []
                for ele in mendeleev.get_all_elements():
                    self.element_types.append(ele.symbol)
            else:
                self.element_types = []
                for ele in mendeleev.get_all_elements():
                    if ele.symbol in element_types:
                        self.element_types.append(ele.symbol)
                self.element_types = element_types
            group_id = self.get_group_ids()
            period = self.get_period()
            covalent_radius = self.get_covalent_radius()
            electron_affinity = self.get_electron_affinity()
            block = self.get_block()
            atomic_volume = self.get_atomic_volume()
            atomic_number = self.get_atomic_number()
            atomic_weight = self.get_atomic_weight()
            electronegativity = self.get_electronegativity()
            valenceelectrons = self.get_valence_electrons()
            ionenergies = self.get_ionenergies()
            for iele, ele in enumerate(self.element_types):
                nfeatures = 0
                self.atom_embeddings[str(mendeleev.element(ele).atomic_number)] = []
                for var in [
                    group_id,
                    period,
                    covalent_radius,
                    electron_affinity,
                    block,
                    atomic_volume,
                    atomic_number,
                    atomic_weight,
                    electronegativity,
                    valenceelectrons,
                    ionenergies,
                ]:
                    nfeatures += var.size()[1]
                    self.atom_embeddings[
                        str(mendeleev.element(ele).atomic_number)
                    ].extend(var[iele, :].tolist())
            with open(embeddingfilename, "w") as f:
                json.dump(self.atom_embeddings, f)

    def get_group_ids(self):
        group_id = []
        for ele in self.element_types:
            group_id.append(self.get_group_id(ele) - 1)
        group_id = F.one_hot(torch.Tensor(group_id).long(), num_classes=-1)
        return group_id

    def get_group_id(self, element):
        if mendeleev.element(element).group_id is not None:
            return mendeleev.element(element).group_id
        else:  # 59-71
            print("Warning: manually assign group_id for element: %s" % element)
            if mendeleev.element(element).atomic_number < 71:
                return mendeleev.element(element).atomic_number - 57 + 3
            elif mendeleev.element(element).atomic_number == 71:
                return 3
            elif mendeleev.element(element).atomic_number < 103:
                return mendeleev.element(element).atomic_number - 89 + 3
            elif mendeleev.element(element).atomic_number == 103:
                return 3
            else:
                raise ValueError(
                    f"unexpected {mendeleev.element(element).atomic_number}"
                )

    def get_period(self):
        period = []
        for ele in self.element_types:
            period.append(mendeleev.element(ele).period - 1)
        period = F.one_hot(torch.Tensor(period).long(), num_classes=-1)
        return period

    def __listocategorical__(self, list, prop_name, num_classes=10):
        list_value = [v for v in list if v is not None]
        if len(list_value) < len(list):
            print(
                f"Warning: None is identified in property, {prop_name}, of some elements. It will be replaced with the minimum value of property for all available elements."
            )
        minval = min(list_value)
        maxval = max(list_value)
        delval = (maxval - minval) / num_classes
        list = [v if v is not None else minval for v in list]
        categories = [
            min(int((item - minval) / delval), num_classes - 1) for item in list
        ]
        return categories

    def get_covalent_radius(self):
        cr = []
        for ele in self.element_types:
            cr.append(mendeleev.element(ele).covalent_radius)
        cr = self.__listocategorical__(cr, "covalent_radius")
        cr = F.one_hot(torch.Tensor(cr).long(), num_classes=10)
        return cr

    def get_atomic_number(self):
        an = []
        for ele in self.element_types:
            an.append(mendeleev.element(ele).atomic_number)
        an = self.__listocategorical__(an, "atomic_number")
        an = F.one_hot(torch.Tensor(an).long(), num_classes=10)
        return an

    def get_atomic_weight(self):
        aw = []
        for ele in self.element_types:
            aw.append(mendeleev.element(ele).atomic_weight)
        aw = self.__listocategorical__(aw, "atomic_weight")
        aw = F.one_hot(torch.Tensor(aw).long(), num_classes=10)
        return aw

    def get_electron_affinity(self):
        ea = []
        for ele in self.element_types:
            ea.append(mendeleev.element(ele).electron_affinity)
        ea = self.__listocategorical__(ea, "electron_affinity")
        ea = F.one_hot(torch.Tensor(ea).long(), num_classes=10)
        return ea

    def get_block(self):
        blocklist = ["s", "p", "d", "f"]
        block = []
        for ele in self.element_types:
            block.append(blocklist.index(mendeleev.element(ele).block))
        block = F.one_hot(torch.Tensor(block).long(), num_classes=-1)
        return block

    def get_atomic_volume(self):
        av = []
        for ele in self.element_types:
            av.append(mendeleev.element(ele).atomic_volume)
        av = self.__listocategorical__(av, "atomic_volume")
        av = F.one_hot(torch.Tensor(av).long(), num_classes=10)
        return av

    def get_electronegativity(self):
        en = []
        for ele in self.element_types:
            en.append(mendeleev.element(ele).en_pauling)
        en = self.__listocategorical__(en, "en_pauling")
        en = F.one_hot(torch.Tensor(en).long(), num_classes=10)
        return en

    def get_valence_electrons(self):
        ve = []
        for ele in self.element_types:
            ve.append(mendeleev.element(ele).nvalence() - 1)
        ve = F.one_hot(torch.Tensor(ve).long(), num_classes=-1)
        return ve

    def get_ionenergies(self):
        ie = []
        for ele in self.element_types:
            degrees = mendeleev.element(ele).ionenergies.keys()
            if len(degrees) == 0:
                ie.append(None)
            else:
                degree = min(degrees)
                ie.append(mendeleev.element(ele).ionenergies[degree])
        ie = self.__listocategorical__(ie, "ionenergies")
        ie = F.one_hot(torch.Tensor(ie).long(), num_classes=10)
        return ie

    def get_atom_features(self, atomtype):
        if isinstance(atomtype, str):
            atomtype = mendeleev.element(atomtype).atomic_number
        return torch.tensor(self.atom_embeddings[str(atomtype)])


if __name__ == "__main__":
    atomicdescriptor = atomicdescriptors(
        "./embedding.json", overwritten=True, element_types=None
    )
    print(atomicdescriptor.get_atom_features("C"))
