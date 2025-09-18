import bz2

import os, json, yaml
import logging
import sys
from mpi4py import MPI
import argparse

import numpy as np

import random

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.transforms import Distance, Spherical, LocalCartesian

import hydragnn
from hydragnn.utils.descriptors_and_embeddings.chemicaldescriptors import (
    ChemicalFeatureEncoder,
)
from hydragnn.utils.descriptors_and_embeddings.topologicaldescriptors import (
    compute_topo_features,
)
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    RadiusGraphPBC,
    PBCDistance,
    PBCLocalCartesian,
    pbc_as_tensor,
)
from hydragnn.preprocess.load_data import split_dataset

import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.utils.print.print_utils import iterate_tqdm, log

from generate_dictionaries_pure_elements import (
    generate_dictionary_bulk_energies,
    generate_dictionary_elements,
)

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import subprocess
from hydragnn.utils.distributed import nsplit
import glob


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def list_directories(path):
    # List all items in the given path
    items = os.listdir(path)

    # Filter out items that are directories
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]

    return directories


periodic_table = generate_dictionary_elements()

# Reversing the dictionary so the elements become keys and the atomic numbers become values
reversed_dict_periodic_table = {value: key for key, value in periodic_table.items()}

# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)

# transform_coordinates_pbc = PBCLocalCartesian(norm=False, cat=False)
transform_coordinates_pbc = PBCDistance(norm=False, cat=False)


class Alexandria(AbstractBaseDataset):
    def __init__(
        self,
        dirpath,
        config,
        graphgps_transform=None,
        energy_per_atom=True,
        dist=False,
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )
        self.radius_graph_pbc = RadiusGraphPBC(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform

        # Threshold for atomic forces in eV/angstrom
        self.forces_norm_threshold = 1000.0

        data_dir = os.path.join(dirpath, "compressed_data", "alexandria.icams.rub.de")
        # print("glob:", os.path.join(data_dir, "**/*.json.bz2"))
        total_file_list = glob.glob(
            os.path.join(data_dir, "**/*.json.bz2"), recursive=True
        )
        if self.dist:
            local_file_list = list(nsplit(total_file_list, self.world_size))[self.rank]
        else:
            local_file_list = total_file_list
        print(
            self.rank,
            "Total flies:",
            len(total_file_list),
            "Local files:",
            len(local_file_list),
        )

        for filepath in iterate_tqdm(local_file_list, verbosity_level=2):
            if filepath.endswith("bz2"):
                self.process_file_content(filepath)
            else:
                print(f"{filepath} is not a .bz2 file to decompress", flush=True)

    def get_data_dict(self, computed_entry_dict):
        """
        Processes the ComputedStructureEntry dictionary to extract the structure, forces and magnetic moments
        and other target properties.
        """

        data_object = None

        def get_forces_array_from_structure(structure):
            forces = [site["properties"]["forces"] for site in structure["sites"]]
            return np.array(forces)

        def get_magmoms_array_from_structure(structure):
            magmoms = [site["properties"]["magmom"] for site in structure["sites"]]
            return np.array(magmoms)

        entry_id = computed_entry_dict["data"]["mat_id"]
        structure = computed_entry_dict["structure"]

        pos = None
        try:
            pos = torch.tensor(
                [item["xyz"] for item in computed_entry_dict["structure"]["sites"]]
            ).to(torch.float32)
            assert pos.shape[1] == 3, "pos tensor does not have 3 coordinates per atom"
            assert pos.shape[0] > 0, "pos tensor does not have any atoms"
        except:
            print(f"Structure {entry_id} does not have positional sites", flush=True)
        natoms = torch.IntTensor([pos.shape[0]])

        cell = None
        try:
            cell = torch.tensor(
                structure["lattice"]["matrix"], dtype=torch.float32
            ).view(3, 3)
        except:
            print(f"Structure {entry_id} does not have cell", flush=True)

        pbc = None
        try:
            pbc = pbc_as_tensor(structure["lattice"]["pbc"])
        except:
            print(f"Structure {entry_id} does not have pbc", flush=True)

        # If either cell or pbc were not read, we set to defaults
        if cell is None or pbc is None:
            cell = torch.eye(3, dtype=torch.float32)
            pbc = torch.tensor([False, False, False], dtype=torch.bool)

        atomic_numbers = None
        try:
            atomic_numbers = (
                torch.tensor(
                    [
                        reversed_dict_periodic_table[item["species"][0]["element"]]
                        for item in computed_entry_dict["structure"]["sites"]
                    ]
                )
                .unsqueeze(1)
                .to(torch.float32)
            )
            assert (
                pos.shape[0] == atomic_numbers.shape[0]
            ), f"pos.shape[0]:{pos.shape[0]} does not match with atomic_numbers.shape[0]:{atomic_numbers.shape[0]}"
        except:
            print(
                f"Structure {entry_id} does not have positional atomic numbers",
                flush=True,
            )
            return data_object

        forces_numpy = None
        try:
            forces_numpy = get_forces_array_from_structure(structure)
        except:
            print(f"Structure {entry_id} does not have forces", flush=True)
            return data_object
        forces = torch.tensor(forces_numpy).to(torch.float32)

        # magmoms_numpy = None
        # try:
        #    magmoms_numpy = get_magmoms_array_from_structure(structure)
        # except:
        #    print(f"Structure {entry_id} does not have magnetic moments")
        #    return data_object

        total_energy = None
        try:
            total_energy = computed_entry_dict["data"]["energy_total"]
        except:
            print(f"Structure {entry_id} does not have total energy", flush=True)
            return data_object
        total_energy_tensor = (
            torch.tensor(total_energy).unsqueeze(0).unsqueeze(1).to(torch.float32)
        )
        total_energy_per_atom_tensor = total_energy_tensor.detach().clone() / natoms

        # total_mag = None
        # try:
        #    total_mag=computed_entry_dict["data"]["total_mag"]
        # except:
        #    print(f"Structure {entry_id} does not have total magnetization")
        #    return data_object

        # dos_ef = None
        # try:
        #    dos_ef=computed_entry_dict["data"]["dos_ef"]
        # except:
        #    print(f"Structure {entry_id} does not have dos_ef")
        #    return data_object

        # band_gap_ind = None
        # try:
        #    band_gap_ind=computed_entry_dict["data"]["band_gap_ind"]
        # except:
        #    print(f"Structure {entry_id} does not have band_gap_ind")
        #    return data_object

        formation_energy = None
        try:
            formation_energy = computed_entry_dict["data"]["e_form"]
        except:
            print(f"Structure {entry_id} does not have formation energy")
            return data_object
        formation_energy_tensor = (
            torch.tensor(formation_energy).unsqueeze(0).unsqueeze(1).to(torch.float32)
        )
        formation_energy_per_atom_tensor = (
            formation_energy_tensor.detach().clone() / natoms
        )

        # energy_above_hull = None
        # try:
        #    energy_above_hull=computed_entry_dict["data"]["e_above_hull"]
        # except:
        #    print(f"Structure {entry_id} does not have e_above_hull")
        #    return data_object

        x = torch.cat([atomic_numbers, pos, forces], dim=1)

        # Calculate chemical composition
        atomic_number_list = atomic_numbers.tolist()
        assert len(atomic_number_list) == natoms
        ## 118: number of atoms in the periodic table
        hist, _ = np.histogram(atomic_number_list, bins=range(1, 118 + 2))
        chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

        data_object = Data(
            dataset_name="alexandria",
            natoms=natoms,
            pos=pos,
            cell=cell,
            pbc=pbc,
            edge_index=None,
            edge_attr=None,
            atomic_numbers=atomic_numbers,
            chemical_composition=chemical_composition,
            smiles_string=None,
            # entry_id=entry_id,
            x=x,
            energy=formation_energy_tensor,
            energy_per_atom=formation_energy_per_atom_tensor,
            forces=forces,
            # formation_energy=torch.tensor(formation_energy).float(),
            # formation_energy_per_atom=torch.tensor(formation_energy_per_atom).float(),
            # energy_above_hull=energy_above_hull,
            # magmoms=torch.tensor(magmoms_numpy).float(),
            # total_mag=total_mag,
            # dos_ef=dos_ef,
            # band_gap_ind=band_gap_ind,
        )

        if self.energy_per_atom:
            data_object.y = data_object.energy_per_atom
        else:
            data_object.y = data_object.energy

        # Apply radius graph and build edge attributes accordingly
        if data_object.pbc.any():
            try:
                data_object = self.radius_graph_pbc(data_object)
                data_object = transform_coordinates_pbc(data_object)
            except:
                print(
                    f"Structure {entry_id} could not successfully apply one or both of the pbc radius graph and positional transform",
                    flush=True,
                )
                data_object = self.radius_graph(data_object)
                data_object = transform_coordinates(data_object)
        else:
            data_object = self.radius_graph(data_object)
            data_object = transform_coordinates(data_object)

        # Default edge_shifts for when radius_graph_pbc is not activated
        if not hasattr(data_object, "edge_shifts"):
            data_object.edge_shifts = torch.zeros(
                (data_object.edge_index.size(1), 3), dtype=torch.float32
            )

        # FIXME: PBC from bool --> int32 to be accepted by ADIOS
        data_object.pbc = data_object.pbc.int()

        # LPE
        if self.graphgps_transform is not None:
            data_object = self.graphgps_transform(data_object)

        # if not data_object:
        #     return None
        # else:
        if self.check_forces_values(data_object.forces):
            return data_object
        else:
            print(
                f"L2-norm of force tensor exceeds threshold {self.forces_norm_threshold} - atomistic structure: {data_object}",
                flush=True,
            )
            return None

    def process_file_content(self, filepath):
        """
        Download a file from a dataset of the Alexandria database with the respective index
        and write it to the LMDB file with the respective index.

        Parameters
        ----------
        filepath : int
            path of file to decompress and open. 3D 0-44, 2D 0-1, 1d 0, scan/pbesol 0-4
        """

        # Open the .bz2 file in binary read mode
        with open(filepath, "rb") as file:
            # Read the compressed data
            compressed_data = file.read()

            # Decompress the data
            try:
                decompressed_data = bz2.decompress(compressed_data)
                json_str = decompressed_data.decode("utf-8")
                data = json.loads(json_str)

                computed_entry_dict = [
                    self.get_data_dict(entry)
                    for entry in iterate_tqdm(
                        data["entries"],
                        desc=f"Rank {self.rank} - Processing file {os.path.basename(filepath)}",
                        verbosity_level=2,
                    )
                ]

                # remove None elements
                filtered_computed_entry_dict = [
                    x for x in computed_entry_dict if x is not None
                ]

                random.shuffle(filtered_computed_entry_dict)
                self.dataset.extend(filtered_computed_entry_dict)

            except OSError as e:
                print(
                    "Failed to decompress data:",
                    e,
                    os.path.basename(filepath),
                    flush=True,
                )
                decompressed_data = None
            except json.JSONDecodeError as e:
                print(
                    "Failed to decode JSON:", e, os.path.basename(filepath), flush=True
                )
            except Exception as e:
                print("An error occurred:", e, os.path.basename(filepath), flush=True)

    def check_forces_values(self, forces):

        # Calculate the L2 norm for each row
        norms = torch.norm(forces, p=2, dim=1)
        # Check if all norms are less than the threshold

        return torch.all(norms < self.forces_norm_threshold).item()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="alexandria_energy.json"
    )
    parser.add_argument(
        "--energy_per_atom",
        help="option to normalize energy by number of atoms",
        type=bool,
        default=False,
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument(
        "--compute_grad_energy", type=bool, help="compute_grad_energy", default=False
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "cartesian_coordinates", "forces"]
    node_feature_dims = [1, 3, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "Alexandria" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "Alexandria" if args.modelname is None else args.modelname
    if args.preonly:
        # Transformation to create positional and structural laplacian encoders
        # Chemical encoder
        ChemEncoder = ChemicalFeatureEncoder()

        # LPE
        lpe_transform = AddLaplacianEigenvectorPE(
            k=config["NeuralNetwork"]["Architecture"]["num_laplacian_eigs"],
            attr_name="lpe",
            is_undirected=True,
        )

        def graphgps_transform(data):
            try:
                data = lpe_transform(data)  # lapPE
            except:
                data.lpe = torch.zeros(
                    [
                        data.num_nodes,
                        config["NeuralNetwork"]["Architecture"]["num_laplacian_eigs"],
                    ],
                    dtype=data.x.dtype,
                    device=data.x.device,
                )
            data = ChemEncoder.compute_chem_features(data)
            data = compute_topo_features(data)
            return data

        ## local data
        total = Alexandria(
            os.path.join(datadir),
            config,
            graphgps_transform=graphgps_transform,
            # graphgps_transform=None,
            energy_per_atom=args.energy_per_atom,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=0.9,
            stratify_splitting=False,
        )
        print(rank, "Local splitting: ", len(trainset), len(valset), len(testset))

        print("Before COMM.Barrier()", flush=True)
        comm.Barrier()
        print("After COMM.Barrier()", flush=True)

        deg = gather_deg(trainset)
        config["pna_deg"] = deg

        setnames = ["trainset", "valset", "testset"]

        ## adios
        if args.format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % modelname
            )
            adwriter = AdiosWriter(fname, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
            # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
            adwriter.add_global("pna_deg", deg)
            adwriter.save()

        ## pickle
        elif args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
            )
            attrs = dict()
            attrs["pna_deg"] = deg
            SimplePickleWriter(
                trainset,
                basedir,
                "trainset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
                attrs=attrs,
            )
            SimplePickleWriter(
                valset,
                basedir,
                "valset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
            SimplePickleWriter(
                testset,
                basedir,
                "testset",
                # minmax_node_feature=total.minmax_node_feature,
                # minmax_graph_feature=total.minmax_graph_feature,
                use_subdir=True,
            )
        sys.exit(0)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    if args.format == "adios":
        info("Adios load")
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        # minmax_node_feature = trainset.minmax_node_feature
        # minmax_graph_feature = trainset.minmax_graph_feature
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            # trainset.minmax_node_feature = minmax_node_feature
            # trainset.minmax_graph_feature = minmax_graph_feature
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity
    )

    # Print details of neural network architecture
    print_model(model)

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
        compute_grad_energy=args.compute_grad_energy,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)
    if writer is not None:
        writer.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
