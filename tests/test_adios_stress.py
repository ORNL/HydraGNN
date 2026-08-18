import torch
from mpi4py import MPI
from torch_geometric.data import Data

from hydragnn.utils.datasets.adiosdataset import AdiosDataset, AdiosWriter


def pytest_adios_stress_round_trip(tmp_path):
    stress = torch.tensor(
        [
            [0.04, 0.01, -0.02],
            [0.01, -0.03, 0.005],
            [-0.02, 0.005, 0.02],
        ],
        dtype=torch.float32,
    )
    samples = [
        Data(dataset_name="stress-test", stress=stress),
        Data(dataset_name="stress-test", stress=-stress),
    ]
    filename = str(tmp_path / "stress.bp")

    writer = AdiosWriter(filename, MPI.COMM_SELF)
    writer.add("trainset", samples)
    writer.save()

    dataset = AdiosDataset(
        filename,
        "trainset",
        MPI.COMM_SELF,
        keys=["stress"],
    )
    assert len(dataset) == 2
    assert dataset[0].stress.shape == (3, 3)
    torch.testing.assert_close(dataset[0].stress, stress)
    torch.testing.assert_close(dataset[1].stress, -stress)
