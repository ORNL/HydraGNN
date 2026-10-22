import pytest
import torch
from torch_geometric.data import Data

pytest.importorskip("adios2")
MPI = pytest.importorskip("mpi4py.MPI")
from hydragnn.data.adiosdataset import AdiosDataset, AdiosMultiDataset, AdiosWriter


@pytest.fixture
def adios_file(tmp_path):
    torch.manual_seed(0)
    samples = [
        Data(
            x=torch.randn(n, 3),
            edge_index=torch.randint(0, n, (2, 2 * n)),
            y=torch.tensor([float(n)]),
            smiles="C" * n,
        )
        for n in (3, 5, 4)
    ]
    filename = str(tmp_path / "graphs.bp")
    writer = AdiosWriter(filename, MPI.COMM_SELF)
    writer.add("trainset", samples)
    writer.save()
    return filename, samples


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"preload": True}, {"shmem": True}, {"enable_cache": True}],
    ids=["file", "preload", "shmem", "cache"],
)
def test_adios_read_modes_round_trip(adios_file, kwargs):
    filename, samples = adios_file
    dataset = AdiosDataset(filename, "trainset", MPI.COMM_SELF, **kwargs)
    try:
        assert len(dataset) == len(samples)
        for i, expected in enumerate(samples):
            sample = dataset[i]
            torch.testing.assert_close(sample.x, expected.x)
            assert torch.equal(sample.edge_index, expected.edge_index)
            assert sample.smiles == expected.smiles
    finally:
        dataset.unlink()


def test_adios_multidataset_concatenates(adios_file):
    filename, samples = adios_file
    dataset = AdiosMultiDataset([filename, filename], "trainset", MPI.COMM_SELF)
    assert len(dataset) == 2 * len(samples)
    torch.testing.assert_close(dataset[len(samples) + 1].x, samples[1].x)
    with pytest.raises(IndexError):
        dataset.get(2 * len(samples))
