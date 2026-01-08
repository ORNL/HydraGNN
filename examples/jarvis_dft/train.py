import bz2

import os, json
import logging
import sys
from mpi4py import MPI
import argparse

import random
import numpy as np

import torch
from torch_geometric.data import Data

from torch_geometric.transforms import Distance, Spherical, LocalCartesian

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.model import print_model
from hydragnn.utils.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset
from hydragnn.preprocess.utils import gather_deg
from hydragnn.preprocess.utils import RadiusGraph, RadiusGraphPBC
from hydragnn.preprocess.load_data import split_dataset

import hydragnn.utils.tracer as tr
from hydragnn.utils.print_utils import iterate_tqdm, log

from generate_dictionaries_pure_elements import (
    generate_dictionary_bulk_energies,
    generate_dictionary_elements,
)

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import subprocess
from hydragnn.utils import nsplit


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


periodic_table = generate_dictionary_elements()

# Reversing the dictionary so the elements become keys and the atomic numbers become values
reversed_dict_periodic_table = {value: key for key, value in periodic_table.items()}

# transform_coordinates = Spherical(norm=False, cat=False)
# transform_coordinates = LocalCartesian(norm=False, cat=False)
transform_coordinates = Distance(norm=False, cat=False)


class JARVIS_DFT(AbstractBaseDataset):
    def __init__(self, dirpath, var_config, energy_per_atom=True, dist=False):
        super().__init__()

        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self.energy_per_atom = energy_per_atom

        self.radius_graph = RadiusGraph(5.0, loop=False, max_num_neighbors=50)

        subdirpath = os.path.join(dirpath, "JARVIS-DFT")

        # List all files in the directory
        all_files = os.listdir(subdirpath)

        # Filter out the files that end with ".json"
        json_files = [
            file
            for file in all_files
            if (file.startswith("jdft") and file.endswith(".json"))
        ]

        for index in json_files:
            self.process_file_content(os.path.join(subdirpath, index))

    def get_data_dict(self, computed_entry_dict):
        """
        Processes the ComputedStructureEntry dictionary to extract the structure, forces and magnetic moments
        and other target properties.
        """

        data_object = None

        def get_forces_array_from_structure(structure):
            forces = [site["properties"]["forces"] for site in structure["sites"]]
            return np.array(forces)

        entry_id = computed_entry_dict["jid"]

        pos = None
        cell = None
        natoms = None
        atomic_numbers = None
        formation_energy_tensor = None

        if "atoms" in computed_entry_dict:
            structure = computed_entry_dict["atoms"]

            try:
                pos = torch.tensor(computed_entry_dict["atoms"]["coords"]).to(
                    torch.float32
                )
                assert (
                    pos.shape[1] == 3
                ), "pos tensor does not have 3 coordinates per atom"
                assert pos.shape[0] > 0, "pos tensor does not have any atoms"
            except:
                print(f"Structure {entry_id} does not have positional sites")
                return data_object
            natoms = torch.IntTensor([pos.shape[0]])

            try:
                cell = torch.tensor(structure["lattice_mat"]).to(torch.float32)
            except:
                print(f"Structure {entry_id} does not have cell")
                return data_object

            try:
                atomic_numbers = (
                    torch.tensor(
                        [
                            [
                                reversed_dict_periodic_table[item]
                                for item in structure["elements"]
                            ]
                        ]
                    )
                    .to(torch.float32)
                    .t()
                )
                assert (
                    pos.shape[0] == atomic_numbers.shape[0]
                ), f"pos.shape[0]:{pos.shape[0]} does not match with atomic_numbers.shape[0]:{atomic_numbers.shape[0]}"
            except:
                print(f"Structure {entry_id} does not have positional atomic numbers")
                return data_object

            try:
                formation_energy_tensor = torch.tensor(
                    computed_entry_dict["formation_energy_peratom"] * natoms
                ).unsqueeze(1)
            except:
                print(f"Structure {entry_id} does not have formation energy per atom")
                return data_object

            formation_energy_per_atom_tensor = (
                torch.tensor(computed_entry_dict["formation_energy_peratom"])
                .unsqueeze(0)
                .unsqueeze(1)
            )

        elif "final_str" in computed_entry_dict:
            try:
                pos = torch.tensor(
                    [item["xyz"] for item in computed_entry_dict["final_str"]["sites"]]
                )
                assert (
                    pos.shape[1] == 3
                ), "pos tensor does not have 3 coordinates per atom"
                assert pos.shape[0] > 0, "pos tensor does not have any atoms"
            except:
                print(f"Structure {entry_id} does not have positional sites")
                return data_object

            try:
                atomic_numbers = torch.tensor(
                    [
                        reversed_dict_periodic_table[item["species"][0]["element"]]
                        for item in computed_entry_dict["final_str"]["sites"]
                    ]
                ).unsqueeze(1)
                assert (
                    pos.shape[0] == atomic_numbers.shape[0]
                ), f"pos.shape[0]:{pos.shape[0]} does not match with atomic_numbers.shape[0]:{atomic_numbers.shape[0]}"
            except:
                return data_object

            try:
                formation_energy_tensor = torch.tensor(
                    computed_entry_dict["final_str"]["formation_energy_peratom"]
                    * natoms
                ).unsqueeze(1)
            except:
                print(f"Structure {entry_id} does not have formation energy per atom")
                return data_object

            formation_energy_per_atom_tensor = (
                torch.tensor(computed_entry_dict["formation_energy_peratom"])
                .unsqueeze(0)
                .unsqueeze(1)
            )

        # total_energy_tensor = None
        # try:
        #     total_energy_tensor = torch.tensor(computed_entry_dict["optb88vdw_total_energy"]).unsqueeze(1)
        # except:
        #     print(f"Structure {entry_id} does not have total energy per atom")
        #     return data_object
        # total_energy_per_atom_tensor = torch.tensor(computed_entry_dict["optb88vdw_total_energy"]).unsqueeze(1)/natoms

        # band_gap = None
        # try:
        #    band_gap=computed_entry_dict["optb88vdw_bandgap"]
        # except:
        #    print(f"Structure {entry_id} does not have band_gap_ind")
        #    return data_object
        # band_gap_tensor = torch.tensor(band_gap_ind).unsqueeze(1)

        # energy_above_hull = None
        # try:
        #    energy_above_hull=computed_entry_dict["ehull"]
        # except:
        #    print(f"Structure {entry_id} does not have e_above_hull")
        #    return data_object
        # energy_above_hull_tensor = torch.tensor(energy_above_hull).unsqueeze(1)

        data_object = Data(
            pos=pos,
            cell=cell,
            atomic_numbers=atomic_numbers,
            # forces=forces,
            # entry_id=entry_id,
            natoms=natoms,
            # total_energy=total_energy_tensor,
            # total_energy_per_atom=total_energy_per_atom_tensor,
            formation_energy=formation_energy_tensor.float(),
            formation_energy_per_atom=formation_energy_per_atom_tensor.float(),
            energy=formation_energy_tensor.float(),
            energy_per_atom=formation_energy_per_atom_tensor.float(),
            # energy_above_hull=energy_above_hull,
            # magmoms=torch.tensor(magmoms_numpy).float(),
            # total_mag=total_mag,
            # dos_ef=dos_ef,
            # band_gap_ind=band_gap_ind,
        )

        if self.energy_per_atom:
            data_object.y = data_object.formation_energy_per_atom
        else:
            data_object.y = data_object.formation_energy

        data_object.x = torch.cat([data_object.atomic_numbers, data_object.pos], dim=1)

        data_object = self.radius_graph(data_object)
        data_object = transform_coordinates(data_object)

        return data_object

    def process_file_content(self, filepath):
        """
        Download a file from a dataset of the Alexandria database with the respective index
        and write it to the LMDB file with the respective index.

        Parameters
        ----------
        filepath : int
            path of JSON file
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(
                    f
                )  # Reads the JSON content from the file and converts it to a Python object
                local_data = list(nsplit(data, self.world_size))[self.rank]

                computed_entry_dict = [
                    self.get_data_dict(entry)
                    for entry in iterate_tqdm(
                        local_data,
                        desc=f"Processing file {filepath}",
                        verbosity_level=2,
                    )
                ]

                # remove None elements
                filtered_computed_entry_dict = [
                    x for x in computed_entry_dict if x is not None
                ]

                random.shuffle(filtered_computed_entry_dict)
                self.dataset.extend(filtered_computed_entry_dict)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON file: {filepath}, error: {str(e)}")
            return None  # Return None if the file is not a valid JSON

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
        "--inputfile", help="input file", type=str, default="jarvis_energy.json"
    )
    parser.add_argument(
        "--energy_per_atom",
        help="option to normalize energy by number of atoms",
        type=bool,
        default=True,
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
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
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "JARVIS_DFT" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "JARVIS_DFT" if args.modelname is None else args.modelname
    if args.preonly:
        ## local data
        total = JARVIS_DFT(
            os.path.join(datadir),
            var_config,
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

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

    hydragnn.utils.save_config(config, log_name)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    # Print details of neural network architecture
    print_model(model)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.load_existing_model_config(
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
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    sys.exit(0)
