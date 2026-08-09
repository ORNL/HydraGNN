import argparse

import torch
from mpi4py import MPI

from hydragnn.utils.datasets.adiosdataset import AdiosDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--validation", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    args = parser.parse_args()

    expected_counts = {
        "trainset": args.train,
        "valset": args.validation,
        "testset": args.test,
    }
    for label, expected_count in expected_counts.items():
        dataset = AdiosDataset(
            args.filename,
            label,
            MPI.COMM_SELF,
            keys=["stress"],
        )
        assert len(dataset) == expected_count, (
            f"{label} contains {len(dataset)} samples, expected {expected_count}"
        )
        for index in {0, len(dataset) // 2, len(dataset) - 1}:
            stress = dataset[index].stress
            assert stress.shape == (3, 3), stress.shape
            assert torch.isfinite(stress).all(), f"Non-finite stress at {label}[{index}]"
            torch.testing.assert_close(stress, stress.T, rtol=1e-5, atol=1e-7)

        print(f"{label}: {len(dataset)} samples with valid stress")


if __name__ == "__main__":
    main()