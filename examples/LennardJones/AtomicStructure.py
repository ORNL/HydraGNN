import torch


class AtomicStructureHandler:

    def __init__(
        self, list_atom_types, bravais_lattice_constants, radius_cutoff, formula
    ):

        self.bravais_lattice_constants = bravais_lattice_constants
        self.radius_cutoff = radius_cutoff
        self.formula = formula

    def compute(self, data):

        assert data.pos.shape[0] == data.x.shape[0]

        interatomic_potential = torch.zeros([data.pos.shape[0], 1])
        interatomic_forces = torch.zeros([data.pos.shape[0], 3])

        for node_id in range(data.pos.shape[0]):

            neighbor_list_indices = torch.where(data.edge_index[0, :] == node_id)[
                0
            ].tolist()
            neighbor_list = data.edge_index[1, neighbor_list_indices]

            for neighbor_id, edge_id in zip(neighbor_list, neighbor_list_indices):

                neighbor_pos = data.pos[neighbor_id, :]
                distance_vector = data.pos[neighbor_id, :] - data.pos[node_id, :]

                # Adjust the neighbor position based on periodic boundary conditions (PBC)
                ## If the distance between the atoms is larger than the cutoff radius, the edge is because of PBC conditions
                if torch.norm(distance_vector) > self.radius_cutoff:
                    ## At this point, we know that the edge is due to PBC conditions, so we need to adjust the neighbor position. We also know that
                    ## that this connection MUST be the closest connection possible as a result of the asserted radius_cutoff < supercell_size earlier
                    ## in the code. Because of this, we can simply adjust the neighbor position coordinate-wise to be closer than
                    ## as done in the following lines of code. The logic goes that if the distance vector[index] is larger than half the supercell size,
                    ## then there is a closer distance at +- supercell_size[index], and we adjust to that for each coordinate
                    if abs(distance_vector[0]) > data.supercell_size[0, 0] / 2:
                        if distance_vector[0] > 0:
                            neighbor_pos[0] -= data.supercell_size[0, 0]
                        else:
                            neighbor_pos[0] += data.supercell_size[0, 0]

                    if abs(distance_vector[1]) > data.supercell_size[1, 1] / 2:
                        if distance_vector[1] > 0:
                            neighbor_pos[1] -= data.supercell_size[1, 1]
                        else:
                            neighbor_pos[1] += data.supercell_size[1, 1]

                    if abs(distance_vector[2]) > data.supercell_size[2, 2] / 2:
                        if distance_vector[2] > 0:
                            neighbor_pos[2] -= data.supercell_size[2, 2]
                        else:
                            neighbor_pos[2] += data.supercell_size[2, 2]

                # The distance vecor may need to be updated after applying PBCs
                distance_vector = data.pos[node_id, :] - neighbor_pos

                # pair_distance = data.edge_attr[edge_id].item()
                interatomic_potential[node_id] += self.formula.potential_energy(
                    distance_vector
                )

                derivative_x = self.formula.derivative_x(distance_vector)
                derivative_y = self.formula.derivative_y(distance_vector)
                derivative_z = self.formula.derivative_z(distance_vector)

                interatomic_forces_contribution_x = -derivative_x
                interatomic_forces_contribution_y = -derivative_y
                interatomic_forces_contribution_z = -derivative_z

                interatomic_forces[node_id, 0] += interatomic_forces_contribution_x
                interatomic_forces[node_id, 1] += interatomic_forces_contribution_y
                interatomic_forces[node_id, 2] += interatomic_forces_contribution_z

        data.x = torch.cat(
            (data.x, interatomic_potential, interatomic_forces),
            1,
        )

        return data
