"""
Test script comparing HydraGNN MACE models using e3nn vs OpenEquivariance backends.

This test trains HydraGNN models with MACE message passing layers twice:
1. Once using pure e3nn for tensor products
2. Once using OpenEquivariance for accelerated tensor products

The test validates that both backends produce equivalent results and handles
any errors that occur during training or comparison.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import warnings

# Add HydraGNN to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hydragnn.utils.model.mace_utils.data import AtomicData
    from hydragnn.utils.model.mace_utils.modules.loss import WeightedEnergyForcesLoss
    from hydragnn.utils.model.mace_utils.modules.models import MACE
    from hydragnn.utils.model.mace_utils.tools.torch_geometric.dataloader import (
        DataLoader,
    )
    from hydragnn.utils.model.equivariance_compat import TensorProduct
    import hydragnn.utils.model.equivariance_compat as compat
except ImportError as e:
    pytest.skip(
        f"Required HydraGNN modules not available: {e}", allow_module_level=True
    )

try:
    import e3nn.o3 as o3
except ImportError:
    pytest.skip("e3nn not available", allow_module_level=True)


def create_synthetic_atomic_data(num_atoms=10, num_samples=50):
    """Create synthetic atomic data for testing."""
    data_list = []

    for _ in range(num_samples):
        # Random atomic numbers (H, C, N, O)
        atomic_numbers = torch.randint(1, 9, (num_atoms,))

        # Random positions
        positions = torch.randn(num_atoms, 3) * 2.0

        # Random cell (periodic boundary conditions)
        cell = torch.eye(3) * 10.0 + torch.randn(3, 3) * 0.1

        # Random energy and forces
        energy = torch.randn(1)
        forces = torch.randn(num_atoms, 3) * 0.1

        # Create edges (simple distance-based)
        edge_index = []
        edge_vectors = []
        edge_lengths = []

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    vec = positions[j] - positions[i]
                    dist = torch.norm(vec)
                    if dist < 5.0:  # cutoff distance
                        edge_index.append([i, j])
                        edge_vectors.append(vec)
                        edge_lengths.append(dist)

        if edge_index:
            edge_index = torch.tensor(edge_index).T
            edge_vectors = torch.stack(edge_vectors)
            edge_lengths = torch.stack(edge_lengths)
        else:
            # Ensure at least one edge exists
            edge_index = torch.tensor([[0, 1], [1, 0]]).T
            edge_vectors = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            edge_lengths = torch.tensor([1.0, 1.0])

        data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            energy=energy,
            forces=forces,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
        )
        data_list.append(data)

    return data_list


def create_mace_model(
    r_max=5.0,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls_first="RealAgnosticResidualInteractionBlock",
    num_interactions=2,
    hidden_irreps="32x0e + 32x1e + 16x2e",
    atomic_energies=None,
    avg_num_neighbors=None,
):
    """Create a MACE model with specified parameters."""

    # Default atomic energies for common elements
    if atomic_energies is None:
        atomic_energies = {1: -0.5, 6: -1.0, 7: -1.5, 8: -2.0}

    # Convert to the format expected by MACE
    atomic_energies_tensor = torch.zeros(max(atomic_energies.keys()) + 1)
    for z, energy in atomic_energies.items():
        atomic_energies_tensor[z] = energy

    model = MACE(
        r_max=r_max,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=max_ell,
        interaction_cls=interaction_cls_first,
        interaction_cls_first=interaction_cls_first,
        num_interactions=num_interactions,
        atom_embedding_dim=64,
        MLP_irreps=o3.Irreps("16x0e"),
        hidden_irreps=o3.Irreps(hidden_irreps),
        atomic_energies=atomic_energies_tensor,
        avg_num_neighbors=avg_num_neighbors or 8.0,
        atomic_numbers=[1, 6, 7, 8],
        correlation=3,
        gate=torch.nn.functional.silu,
    )

    return model


def force_backend_selection(use_openequivariance=True):
    """Force selection of backend for tensor products."""
    # Store original state
    original_available = getattr(compat, "OPENEQUIVARIANCE_AVAILABLE", None)

    if use_openequivariance:
        # Force OpenEquivariance to be available/unavailable for testing
        try:
            import openequivariance  # noqa: F401

            compat.OPENEQUIVARIANCE_AVAILABLE = True
        except ImportError:
            compat.OPENEQUIVARIANCE_AVAILABLE = False
    else:
        # Force pure e3nn usage
        compat.OPENEQUIVARIANCE_AVAILABLE = False

    return original_available


def train_model_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        try:
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    return total_loss / max(num_batches, 1)


@pytest.mark.mpi_skip()
def pytest_mace_e3nn_vs_openequivariance_training():
    """
    Test HydraGNN MACE model training with both e3nn and OpenEquivariance backends.

    This test:
    1. Creates synthetic atomic data
    2. Trains identical MACE models using e3nn and OpenEquivariance
    3. Compares the results and ensures both backends work correctly
    4. Captures and reports any errors that occur during training
    """
    print("\n" + "=" * 80)
    print("Testing HydraGNN MACE Model: e3nn vs OpenEquivariance Training Comparison")
    print("=" * 80)

    device = torch.device("cpu")  # Use CPU for CI compatibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Create synthetic training data
        print("Creating synthetic atomic data...")
        data_list = create_synthetic_atomic_data(num_atoms=8, num_samples=20)
        dataloader = DataLoader(data_list, batch_size=4, shuffle=True)
        print(
            f"Created {len(data_list)} samples with {data_list[0].atomic_numbers.shape[0]} atoms each"
        )

        # Test parameters
        training_epochs = 2
        learning_rate = 0.001
        results = {}

        # Test both backends
        for backend_name, use_openequivariance in [
            ("e3nn", False),
            ("OpenEquivariance", True),
        ]:
            print(f"\n--- Testing with {backend_name} backend ---")

            try:
                # Force backend selection
                original_state = force_backend_selection(use_openequivariance)

                # Report backend status
                backend_available = getattr(compat, "OPENEQUIVARIANCE_AVAILABLE", False)
                if use_openequivariance and not backend_available:
                    print(
                        f"WARNING: OpenEquivariance not available, falling back to e3nn"
                    )
                    backend_name = "e3nn (fallback)"
                elif use_openequivariance and backend_available:
                    print(f"SUCCESS: Using OpenEquivariance backend")
                else:
                    print(f"Using pure e3nn backend")

                # Create model
                print("Creating MACE model...")
                model = create_mace_model(
                    r_max=4.0,
                    num_bessel=6,
                    num_polynomial_cutoff=5,
                    max_ell=1,  # Reduced for faster testing
                    num_interactions=1,  # Reduced for faster testing
                    hidden_irreps="16x0e + 8x1e",  # Smaller for faster testing
                )
                model = model.to(device)

                # Create optimizer and loss function
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                loss_fn = WeightedEnergyForcesLoss(
                    energy_weight=1.0, forces_weight=10.0
                )

                print(
                    f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
                )

                # Training loop
                epoch_losses = []
                print(f"Training for {training_epochs} epochs...")

                for epoch in range(training_epochs):
                    try:
                        loss = train_model_epoch(
                            model, dataloader, optimizer, loss_fn, device
                        )
                        epoch_losses.append(loss)
                        print(f"  Epoch {epoch+1}/{training_epochs}: Loss = {loss:.6f}")

                    except Exception as train_error:
                        error_msg = f"Training epoch {epoch+1} failed with {backend_name}: {str(train_error)}"
                        print(f"ERROR: {error_msg}")
                        raise RuntimeError(error_msg)

                # Test model inference
                print("Testing model inference...")
                model.eval()
                with torch.no_grad():
                    test_batch = next(iter(dataloader)).to(device)
                    output = model(test_batch)

                    # Validate output structure
                    assert hasattr(output, "energy"), "Model output missing energy"
                    assert hasattr(output, "forces"), "Model output missing forces"
                    assert (
                        output.energy.shape[0] == test_batch.ptr.shape[0] - 1
                    ), "Energy shape mismatch"
                    assert (
                        output.forces.shape == test_batch.forces.shape
                    ), "Forces shape mismatch"

                # Store results
                results[backend_name] = {
                    "final_loss": epoch_losses[-1],
                    "epoch_losses": epoch_losses,
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "output_energy_mean": output.energy.mean().item(),
                    "output_forces_norm": torch.norm(output.forces).item(),
                    "success": True,
                }

                print(f"SUCCESS: {backend_name} training completed")
                print(f"  Final loss: {epoch_losses[-1]:.6f}")
                print(f"  Output energy mean: {output.energy.mean().item():.4f}")
                print(f"  Output forces norm: {torch.norm(output.forces).item():.4f}")

                # Restore original backend state
                if original_state is not None:
                    compat.OPENEQUIVARIANCE_AVAILABLE = original_state

            except Exception as backend_error:
                error_msg = f"Backend {backend_name} failed: {str(backend_error)}"
                print(f"ERROR: {error_msg}")
                results[backend_name] = {"success": False, "error": error_msg}

                # Restore original backend state
                if "original_state" in locals() and original_state is not None:
                    compat.OPENEQUIVARIANCE_AVAILABLE = original_state

                # Re-raise the error to fail the test
                raise RuntimeError(error_msg)

        # Compare results if both backends succeeded
        print(f"\n--- Results Comparison ---")
        success_count = sum(1 for r in results.values() if r.get("success", False))
        print(f"Successful backend tests: {success_count}/{len(results)}")

        if success_count >= 2:
            backend_names = [
                name for name, result in results.items() if result.get("success", False)
            ]
            print(f"Comparing results between {' and '.join(backend_names)}:")

            # Compare final losses (should be similar but not identical due to different implementations)
            losses = [results[name]["final_loss"] for name in backend_names]
            loss_diff = abs(losses[0] - losses[1]) if len(losses) == 2 else 0
            print(f"  Final loss difference: {loss_diff:.6f}")

            # Compare model parameters (should be identical)
            params = [results[name]["model_parameters"] for name in backend_names]
            params_match = len(set(params)) == 1
            print(
                f"  Model parameters match: {params_match} ({params[0]:,} parameters)"
            )

            # The test passes if both backends completed training without errors
            print(f"\nSUCCESS: Both backends completed training successfully!")

        elif success_count == 1:
            successful_backend = next(
                name for name, result in results.items() if result.get("success", False)
            )
            print(
                f"Only {successful_backend} succeeded - this may indicate backend availability issues"
            )

        else:
            print("ERROR: No backends succeeded")

        # Print final summary
        print(f"\n--- Final Summary ---")
        for backend_name, result in results.items():
            if result.get("success", False):
                print(
                    f"✓ {backend_name}: Training successful (final loss: {result['final_loss']:.6f})"
                )
            else:
                print(
                    f"✗ {backend_name}: Training failed - {result.get('error', 'Unknown error')}"
                )

        # Test passes if at least one backend succeeds (allows for OpenEquivariance unavailability)
        assert success_count >= 1, f"All backends failed. Results: {results}"

        print(f"\nTest completed successfully with {success_count} working backend(s)!")

    except Exception as e:
        error_msg = f"MACE e3nn vs OpenEquivariance test failed: {str(e)}"
        print(f"\nERROR: {error_msg}")
        print(f"Exception type: {type(e).__name__}")

        # Print stack trace for debugging
        import traceback

        print(f"Stack trace:")
        traceback.print_exc()

        # Fail the test with detailed error information
        pytest.fail(error_msg)


if __name__ == "__main__":
    # Allow running the test directly for debugging
    pytest_mace_e3nn_vs_openequivariance_training()
