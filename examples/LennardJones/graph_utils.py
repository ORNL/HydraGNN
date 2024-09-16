import torch
from torch_geometric.transforms import RadiusGraph
import ase
import ase.neighborlist
from torch_geometric.utils import remove_self_loops, degree

class RadiusGraphPBC(RadiusGraph):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance, including periodic images.
    """

    def __call__(self, data):
        data.edge_attr = None
        assert (
            "batch" not in data
        ), "Periodic boundary conditions not currently supported on batches."
        assert hasattr(
            data, "supercell_size"
        ), "The data must contain the size of the supercell to apply periodic boundary conditions."
        ase_atom_object = ase.Atoms(
            positions=data.pos,
            cell=data.supercell_size,
            pbc=True,
        )
        # ‘i’ : first atom index
        # ‘j’ : second atom index
        # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
        edge_src, edge_dst, edge_length = ase.neighborlist.neighbor_list(
            "ijd", a=ase_atom_object, cutoff=self.r, self_interaction=self.loop
        )
        data.edge_index = torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        )

        # ensure no duplicate edges
        num_edges = data.edge_index.size(1)
        data.coalesce()
        assert num_edges == data.edge_index.size(
            1
        ), "Adding periodic boundary conditions would result in duplicate edges. Cutoff radius must be reduced or system size increased."

        data.edge_attr = torch.tensor(edge_length, dtype=torch.float).unsqueeze(1)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"


def get_radius_graph_pbc(radius, max_neighbours, loop=False):
    return RadiusGraphPBC(
        r=radius,
        loop=loop,
        max_num_neighbors=max_neighbours,
    )