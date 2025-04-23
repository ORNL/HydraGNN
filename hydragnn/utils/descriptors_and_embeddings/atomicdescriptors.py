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
        one_hot=False,
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
            self.one_hot = one_hot
            type_id = self.get_type_ids()
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
            if self.one_hot:
                # properties with integer values
                group_id = self.convert_integerproperty_onehot(group_id, num_classes=-1)
                period = self.convert_integerproperty_onehot(period, num_classes=-1)
                atomic_number = self.convert_integerproperty_onehot(
                    atomic_number, num_classes=-1
                )
                valenceelectrons = self.convert_integerproperty_onehot(
                    valenceelectrons, num_classes=-1
                )
                # properties with real values
                covalent_radius = self.convert_realproperty_onehot(
                    covalent_radius, num_classes=10
                )
                electron_affinity = self.convert_realproperty_onehot(
                    electron_affinity, num_classes=10
                )
                atomic_volume = self.convert_realproperty_onehot(
                    atomic_volume, num_classes=10
                )
                atomic_weight = self.convert_realproperty_onehot(
                    atomic_weight, num_classes=10
                )
                electronegativity = self.convert_realproperty_onehot(
                    electronegativity, num_classes=10
                )
                ionenergies = self.convert_realproperty_onehot(
                    ionenergies, num_classes=10
                )

            for iele, ele in enumerate(self.element_types):
                nfeatures = 0
                self.atom_embeddings[str(mendeleev.element(ele).atomic_number)] = []
                for var in [
                    type_id,
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

    def get_type_ids(self):
        type_id = F.one_hot(torch.arange(len(self.element_types)), num_classes=-1)
        return type_id

    def get_group_ids(self, num_classes=-1):
        group_id = []
        for ele in self.element_types:
            group_id.append(self.get_group_id(ele) - 1)
        return torch.Tensor(group_id).reshape(len(self.element_types), -1)

    def get_group_id(self, element):
        if mendeleev.element(element).group_id is not None:
            return mendeleev.element(element).group_id
        else:
            raise ValueError(
                f"None is returned by Mendeleev for group_id of element: {element}"
            )

    def get_period(self, num_classes=-1):
        period = []
        for ele in self.element_types:
            period.append(mendeleev.element(ele).period - 1)
        return torch.Tensor(period).reshape(len(self.element_types), -1)

    def __propertynormalize__(self, prop_list, prop_name):

        None_elements = [
            ele for ele, item in zip(self.element_types, prop_list) if item is None
        ]
        if len(None_elements) > 0:
            raise ValueError(
                f"None is identified in Mendeleev property, {prop_name}, of elements, {None_elements}"
            )

        minval = min(prop_list)
        maxval = max(prop_list)
        return [(item - minval) / (maxval - minval) for item in prop_list]

    def __realtocategorical__(self, prop_tensor, num_classes=10):

        delval = (prop_tensor.max() - prop_tensor.min()) / num_classes
        categories = torch.minimum(
            (prop_tensor - prop_tensor.min()) / delval, torch.tensor([num_classes - 1])
        )
        return categories

    def get_covalent_radius(self):
        cr = []
        for ele in self.element_types:
            cr.append(mendeleev.element(ele).covalent_radius)
        cr = self.__propertynormalize__(cr, "covalent_radius")
        return torch.Tensor(cr).reshape(len(self.element_types), -1)

    def get_atomic_number(self):
        an = []
        for ele in self.element_types:
            an.append(mendeleev.element(ele).atomic_number)
        return torch.Tensor(an).reshape(len(self.element_types), -1)

    def get_atomic_weight(self):
        aw = []
        for ele in self.element_types:
            aw.append(mendeleev.element(ele).atomic_weight)
        aw = self.__propertynormalize__(aw, "atomic_weight")
        return torch.Tensor(aw).reshape(len(self.element_types), -1)

    def get_electron_affinity(self):
        ea = []
        for ele in self.element_types:
            ea.append(mendeleev.element(ele).electron_affinity)
        ea = self.__propertynormalize__(ea, "electron_affinity")
        return torch.Tensor(ea).reshape(len(self.element_types), -1)

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
        av = self.__propertynormalize__(av, "atomic_volume")
        return torch.Tensor(av).reshape(len(self.element_types), -1)

    def get_electronegativity(self):
        en = []
        for ele in self.element_types:
            en.append(mendeleev.element(ele).en_pauling)
        en = self.__propertynormalize__(en, "en_pauling")
        return torch.Tensor(en).reshape(len(self.element_types), -1)

    def get_valence_electrons(self):
        ve = []
        for ele in self.element_types:
            ve.append(mendeleev.element(ele).nvalence())
        return torch.Tensor(ve).reshape(len(self.element_types), -1)

    def get_ionenergies(self):
        ie = []
        for ele in self.element_types:
            degrees = mendeleev.element(ele).ionenergies.keys()
            if len(degrees) == 0:
                ie.append(None)
            else:
                degree = min(degrees)
                ie.append(mendeleev.element(ele).ionenergies[degree])
        ie = self.__propertynormalize__(ie, "ionenergies")
        return torch.Tensor(ie).reshape(len(self.element_types), -1)

    def convert_integerproperty_onehot(self, prop, num_classes=-1):
        prop = F.one_hot(prop.to(torch.int64).squeeze(), num_classes=num_classes)
        return prop

    def convert_realproperty_onehot(self, prop, num_classes=10):
        prop = self.__realtocategorical__(prop, num_classes=num_classes)
        prop = F.one_hot(prop.to(torch.int64).squeeze(), num_classes=num_classes)
        return prop

    def get_atom_features(self, atomtype):
        if isinstance(atomtype, str):
            atomtype = mendeleev.element(atomtype).atomic_number
        return torch.tensor(self.atom_embeddings[str(atomtype)])


if __name__ == "__main__":
    atomicdescriptor = atomicdescriptors(
        "./embedding.json", overwritten=True, element_types=["C", "H", "S"]
    )
    print(atomicdescriptor.get_atom_features("C"))
    print(len(atomicdescriptor.get_atom_features("C")))
    atomicdescriptor_onehot = atomicdescriptors(
        "./embedding_onehot.json",
        overwritten=True,
        element_types=["C", "H", "S"],
        one_hot=True,
    )
    print(atomicdescriptor_onehot.get_atom_features("C"))
    print(len(atomicdescriptor_onehot.get_atom_features("C")))
