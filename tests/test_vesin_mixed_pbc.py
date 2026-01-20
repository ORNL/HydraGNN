"""
Test cases for vesin vs ASE neighbor list behavior with mixed periodic boundary conditions.

Tests:
1. Compare ASE (native mixed PBC) vs vesin (with large-cell workaround) - should match
2. Test vesin's behavior when given mixed PBC directly (without workaround)

Run: python test_vesin_mixed_pbc.py
"""

import numpy as np
import ase
import ase.neighborlist
import vesin
from typing import Tuple, Dict, Any


def create_2d_slab(n_atoms: int = 20, vacuum: float = 15.0, seed: int = 42) -> ase.Atoms:
    """
    Create a 2D periodic slab (periodic in x, y; non-periodic in z).
    
    Atoms are placed in a thin layer with vacuum above/below.
    """
    np.random.seed(seed)
    
    # Cell: periodic in xy, with vacuum in z
    cell = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, vacuum * 2],  # Large z to hold vacuum
    ])
    
    # Place atoms in a thin slab near z=0
    positions = np.random.rand(n_atoms, 3)
    positions[:, 0] *= cell[0, 0]  # x in [0, 10]
    positions[:, 1] *= cell[1, 1]  # y in [0, 10]
    positions[:, 2] = positions[:, 2] * 2.0 + vacuum - 1.0  # z in thin layer around center
    
    atoms = ase.Atoms(positions=positions, cell=cell, pbc=[True, True, False])
    return atoms


def create_1d_chain(n_atoms: int = 15, vacuum: float = 15.0, seed: int = 42) -> ase.Atoms:
    """
    Create a 1D periodic chain (periodic in x only).
    """
    np.random.seed(seed)
    
    cell = np.array([
        [10.0, 0.0, 0.0],
        [0.0, vacuum * 2, 0.0],
        [0.0, 0.0, vacuum * 2],
    ])
    
    positions = np.random.rand(n_atoms, 3)
    positions[:, 0] *= cell[0, 0]
    positions[:, 1] = positions[:, 1] * 2.0 + vacuum - 1.0
    positions[:, 2] = positions[:, 2] * 2.0 + vacuum - 1.0
    
    atoms = ase.Atoms(positions=positions, cell=cell, pbc=[True, False, False])
    return atoms


def create_mixed_pbc_system(pbc: list, n_atoms: int = 20, seed: int = 42) -> ase.Atoms:
    """
    Create a system with arbitrary mixed PBC for testing.
    """
    np.random.seed(seed)
    
    cell = np.array([
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    
    positions = np.random.rand(n_atoms, 3) * 8.0
    atoms = ase.Atoms(positions=positions, cell=cell, pbc=pbc)
    return atoms


def expand_cell_for_vesin(atoms: ase.Atoms, cutoff: float = 10.0) -> ase.Atoms:
    """
    Apply the mixed-PBC workaround: expand cell in non-periodic directions.
    
    Returns a new Atoms object with pbc=[True, True, True] and modified cell.
    
    The expansion uses a geometry-based value to avoid extreme aspect ratios
    that can cause floating point exceptions in vesin (crashes around 2500:1).
    """
    cell = atoms.cell.array.copy()
    pbc = atoms.pbc.tolist()
    pos = atoms.positions
    
    if all(pbc) or not any(pbc):
        # No modification needed
        return atoms.copy()
    
    for i, is_periodic in enumerate(pbc):
        if not is_periodic:
            # Use a value that's large enough to prevent PBC neighbors
            # but not so large it causes numerical issues
            # Value = 2 * (position extent in this direction) + 4 * cutoff
            pos_min = pos[:, i].min()
            pos_max = pos[:, i].max()
            extent = pos_max - pos_min
            large_value = max(extent * 2 + cutoff * 4, 100.0)
            
            cell_vec = cell[i]
            vec_norm = np.linalg.norm(cell_vec)
            if vec_norm > 1e-10:
                cell[i] = cell_vec / vec_norm * large_value
            else:
                cell[i] = np.zeros(3)
                cell[i, i] = large_value
    
    atoms_modified = atoms.copy()
    atoms_modified.set_cell(cell)
    atoms_modified.set_pbc([True, True, True])
    return atoms_modified


def get_ase_neighbor_list(atoms: ase.Atoms, cutoff: float) -> Tuple[np.ndarray, ...]:
    """Get neighbor list using ASE."""
    i, j, d, S = ase.neighborlist.neighbor_list("ijdS", atoms, cutoff)
    return i, j, d, S


def get_vesin_neighbor_list(atoms: ase.Atoms, cutoff: float) -> Tuple[np.ndarray, ...]:
    """Get neighbor list using vesin."""
    i, j, S = vesin.ase_neighbor_list("ijS", a=atoms, cutoff=cutoff)
    # Compute distances manually
    pos = atoms.positions
    cell = atoms.cell.array
    vec = pos[j] - pos[i] + S @ cell
    d = np.linalg.norm(vec, axis=1)
    return i, j, d, S


def compare_neighbor_lists(
    i1: np.ndarray, j1: np.ndarray, d1: np.ndarray, S1: np.ndarray,
    i2: np.ndarray, j2: np.ndarray, d2: np.ndarray, S2: np.ndarray,
    atol: float = 1e-6
) -> Dict[str, Any]:
    """
    Compare two neighbor lists for equivalence.
    
    Returns a dict with comparison results.
    """
    # Create sets of (i, j, S) tuples for comparison
    # Note: S needs to be converted to tuple for hashing
    def make_edge_set(i, j, S):
        return set((int(ii), int(jj), tuple(ss)) for ii, jj, ss in zip(i, j, S))
    
    edges1 = make_edge_set(i1, j1, S1)
    edges2 = make_edge_set(i2, j2, S2)
    
    # Find differences
    only_in_1 = edges1 - edges2
    only_in_2 = edges2 - edges1
    common = edges1 & edges2
    
    # For common edges, compare distances
    dist_match = True
    max_dist_diff = 0.0
    
    if len(common) > 0:
        # Build lookup for distances
        def make_dist_dict(i, j, S, d):
            return {(int(ii), int(jj), tuple(ss)): dd 
                    for ii, jj, ss, dd in zip(i, j, S, d)}
        
        dist1 = make_dist_dict(i1, j1, S1, d1)
        dist2 = make_dist_dict(i2, j2, S2, d2)
        
        for edge in common:
            diff = abs(dist1[edge] - dist2[edge])
            max_dist_diff = max(max_dist_diff, diff)
            if diff > atol:
                dist_match = False
    
    return {
        "n_edges_1": len(edges1),
        "n_edges_2": len(edges2),
        "n_common": len(common),
        "n_only_in_1": len(only_in_1),
        "n_only_in_2": len(only_in_2),
        "edges_match": len(only_in_1) == 0 and len(only_in_2) == 0,
        "distances_match": dist_match,
        "max_distance_diff": max_dist_diff,
        "only_in_1": list(only_in_1)[:5],  # Sample for debugging
        "only_in_2": list(only_in_2)[:5],
    }


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


# =============================================================================
# TEST 1: Compare ASE (native mixed PBC) vs vesin (with workaround)
# =============================================================================

def test_2d_slab_comparison():
    """Test that vesin with large-cell workaround matches ASE for 2D slab."""
    print_separator("TEST 1a: 2D Slab (pbc=[True, True, False])")
    
    cutoff = 5.0
    atoms = create_2d_slab(n_atoms=30, vacuum=15.0)
    
    print(f"System: {len(atoms)} atoms, cell diag = {np.diag(atoms.cell.array)}")
    print(f"PBC: {atoms.pbc.tolist()}")
    print(f"Cutoff: {cutoff} Å")
    
    # ASE with native mixed PBC
    print("\nRunning ASE (native mixed PBC)...")
    i_ase, j_ase, d_ase, S_ase = get_ase_neighbor_list(atoms, cutoff)
    print(f"  ASE found {len(i_ase)} edges")
    
    # Vesin with workaround
    print("\nRunning vesin (with large-cell workaround)...")
    atoms_expanded = expand_cell_for_vesin(atoms, cutoff=cutoff)
    print(f"  Expanded cell diag = {np.diag(atoms_expanded.cell.array)[:2]}... (truncated)")
    i_vesin, j_vesin, d_vesin, S_vesin = get_vesin_neighbor_list(atoms_expanded, cutoff)
    print(f"  vesin found {len(i_vesin)} edges")
    
    # Compare
    result = compare_neighbor_lists(i_ase, j_ase, d_ase, S_ase,
                                     i_vesin, j_vesin, d_vesin, S_vesin)
    
    print("\nComparison:")
    print(f"  Edges match: {result['edges_match']}")
    print(f"  Common edges: {result['n_common']}")
    print(f"  Only in ASE: {result['n_only_in_1']}")
    print(f"  Only in vesin: {result['n_only_in_2']}")
    print(f"  Max distance diff: {result['max_distance_diff']:.2e}")
    
    if result['n_only_in_1'] > 0:
        print(f"  Sample edges only in ASE: {result['only_in_1']}")
    if result['n_only_in_2'] > 0:
        print(f"  Sample edges only in vesin: {result['only_in_2']}")
    
    return result['edges_match'] and result['distances_match']


def test_1d_chain_comparison():
    """Test that vesin with large-cell workaround matches ASE for 1D chain."""
    print_separator("TEST 1b: 1D Chain (pbc=[True, False, False])")
    
    cutoff = 4.0
    atoms = create_1d_chain(n_atoms=20, vacuum=15.0)
    
    print(f"System: {len(atoms)} atoms, cell diag = {np.diag(atoms.cell.array)}")
    print(f"PBC: {atoms.pbc.tolist()}")
    print(f"Cutoff: {cutoff} Å")
    
    # ASE with native mixed PBC
    print("\nRunning ASE (native mixed PBC)...")
    i_ase, j_ase, d_ase, S_ase = get_ase_neighbor_list(atoms, cutoff)
    print(f"  ASE found {len(i_ase)} edges")
    
    # Vesin with workaround
    print("\nRunning vesin (with large-cell workaround)...")
    atoms_expanded = expand_cell_for_vesin(atoms, cutoff=cutoff)
    i_vesin, j_vesin, d_vesin, S_vesin = get_vesin_neighbor_list(atoms_expanded, cutoff)
    print(f"  vesin found {len(i_vesin)} edges")
    
    # Compare
    result = compare_neighbor_lists(i_ase, j_ase, d_ase, S_ase,
                                     i_vesin, j_vesin, d_vesin, S_vesin)
    
    print("\nComparison:")
    print(f"  Edges match: {result['edges_match']}")
    print(f"  Common edges: {result['n_common']}")
    print(f"  Only in ASE: {result['n_only_in_1']}")
    print(f"  Only in vesin: {result['n_only_in_2']}")
    
    return result['edges_match'] and result['distances_match']


def test_various_mixed_pbc():
    """Test various mixed PBC configurations."""
    print_separator("TEST 1c: Various Mixed PBC Configurations")
    
    cutoff = 4.0
    pbc_configs = [
        [True, False, True],   # Periodic in x and z
        [False, True, True],   # Periodic in y and z
        [False, True, False],  # Periodic in y only
        [False, False, True],  # Periodic in z only
    ]
    
    all_passed = True
    
    for pbc in pbc_configs:
        print(f"\n--- Testing pbc={pbc} ---")
        atoms = create_mixed_pbc_system(pbc, n_atoms=25)
        
        # ASE
        i_ase, j_ase, d_ase, S_ase = get_ase_neighbor_list(atoms, cutoff)
        
        # Vesin with workaround
        atoms_expanded = expand_cell_for_vesin(atoms, cutoff=cutoff)
        i_vesin, j_vesin, d_vesin, S_vesin = get_vesin_neighbor_list(atoms_expanded, cutoff)
        
        result = compare_neighbor_lists(i_ase, j_ase, d_ase, S_ase,
                                         i_vesin, j_vesin, d_vesin, S_vesin)
        
        passed = result['edges_match'] and result['distances_match']
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: ASE={result['n_edges_1']} edges, vesin={result['n_edges_2']} edges, "
              f"diff={result['n_only_in_1']}+{result['n_only_in_2']}")
    
    return all_passed


# =============================================================================
# TEST 2: What happens when vesin gets mixed PBC directly (no workaround)?
# =============================================================================

def test_vesin_direct_mixed_pbc():
    """Test vesin's behavior when given mixed PBC directly without workaround."""
    print_separator("TEST 2: vesin with Mixed PBC Directly (No Workaround)")
    
    cutoff = 4.0
    pbc_configs = [
        [True, True, False],
        [True, False, True],
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]
    
    print("Testing what happens when vesin receives mixed PBC directly...\n")
    print("This tests if vesin's native mixed PBC handling matches ASE.\n")
    
    all_match = True
    
    for pbc in pbc_configs:
        print(f"--- pbc={pbc} ---")
        atoms = create_mixed_pbc_system(pbc, n_atoms=15)
        
        try:
            # vesin direct (no workaround)
            i_vesin, j_vesin, S_vesin = vesin.ase_neighbor_list("ijS", a=atoms, cutoff=cutoff)
            pos = atoms.positions
            cell = atoms.cell.array
            vec = pos[j_vesin] - pos[i_vesin] + S_vesin @ cell
            d_vesin = np.linalg.norm(vec, axis=1)
            
            print(f"  vesin returned {len(i_vesin)} edges (no error)")
            
            # Compare with ASE
            i_ase, j_ase, d_ase, S_ase = get_ase_neighbor_list(atoms, cutoff)
            
            result = compare_neighbor_lists(
                i_ase, j_ase, d_ase, S_ase,
                i_vesin, j_vesin, d_vesin, S_vesin
            )
            
            if result['edges_match']:
                print(f"  ✓ Edges MATCH ASE ({len(i_ase)} edges)")
            else:
                print(f"  ⚠ Edges DIFFER from ASE:")
                print(f"    ASE: {result['n_edges_1']} edges")
                print(f"    vesin: {result['n_edges_2']} edges")
                print(f"    Only in ASE: {result['n_only_in_1']}")
                print(f"    Only in vesin: {result['n_only_in_2']}")
                all_match = False
            
            # Also check if it matches pbc=[True,True,True] behavior
            atoms_full_pbc = atoms.copy()
            atoms_full_pbc.set_pbc([True, True, True])
            i_full, j_full, S_full = vesin.ase_neighbor_list("ijS", a=atoms_full_pbc, cutoff=cutoff)
            
            if len(i_vesin) == len(i_full):
                # Same count - might be treating as fully periodic
                # Do full comparison
                vec_full = pos[j_full] - pos[i_full] + S_full @ cell
                d_full = np.linalg.norm(vec_full, axis=1)
                full_result = compare_neighbor_lists(
                    i_vesin, j_vesin, d_vesin, S_vesin,
                    i_full, j_full, d_full, S_full
                )
                if full_result['edges_match']:
                    print(f"  ⚠ Behavior IDENTICAL to pbc=[True,True,True] - vesin may ignore mixed PBC")
            
        except Exception as e:
            print(f"  ✗ vesin raised an error: {type(e).__name__}: {e}")
        
        print()
    
    if all_match:
        print("Conclusion: vesin handles mixed PBC directly and matches ASE!")
        print("The workaround may be unnecessary for your vesin version.")
    else:
        print("Conclusion: vesin's mixed PBC differs from ASE in some cases.")
        print("The workaround is recommended for consistent behavior.")


# =============================================================================
# TEST 3: Edge cases and stress tests
# =============================================================================

def test_edge_cases():
    """Test edge cases like atoms near boundaries, small cells, etc."""
    print_separator("TEST 3: Edge Cases")
    
    cutoff = 5.0
    all_passed = True
    
    # Case 1: Atoms very close to the non-periodic boundary
    print("\n--- Case 1: Atoms near non-periodic boundary ---")
    atoms = ase.Atoms(
        positions=[
            [5.0, 5.0, 0.5],    # Near bottom
            [5.0, 5.0, 29.5],   # Near top
            [5.0, 5.0, 15.0],   # Middle
        ],
        cell=[10.0, 10.0, 30.0],
        pbc=[True, True, False]
    )
    
    i_ase, j_ase, d_ase, S_ase = get_ase_neighbor_list(atoms, cutoff)
    
    atoms_exp = expand_cell_for_vesin(atoms, cutoff=cutoff)
    i_vesin, j_vesin, d_vesin, S_vesin = get_vesin_neighbor_list(atoms_exp, cutoff)
    
    result = compare_neighbor_lists(i_ase, j_ase, d_ase, S_ase,
                                     i_vesin, j_vesin, d_vesin, S_vesin)
    passed = result['edges_match']
    all_passed = all_passed and passed
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: ASE={result['n_edges_1']}, vesin={result['n_edges_2']}")
    
    # Case 2: Very small periodic cell (atoms should wrap around)
    print("\n--- Case 2: Small periodic cell with wrapping ---")
    atoms = ase.Atoms(
        positions=[
            [0.5, 0.5, 15.0],
            [2.5, 0.5, 15.0],
        ],
        cell=[3.0, 3.0, 30.0],
        pbc=[True, True, False]
    )
    
    i_ase, j_ase, d_ase, S_ase = get_ase_neighbor_list(atoms, cutoff)
    
    atoms_exp = expand_cell_for_vesin(atoms, cutoff=cutoff)
    i_vesin, j_vesin, d_vesin, S_vesin = get_vesin_neighbor_list(atoms_exp, cutoff)
    
    result = compare_neighbor_lists(i_ase, j_ase, d_ase, S_ase,
                                     i_vesin, j_vesin, d_vesin, S_vesin)
    passed = result['edges_match']
    all_passed = all_passed and passed
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: ASE={result['n_edges_1']}, vesin={result['n_edges_2']}")
    if not passed:
        print(f"    Only in ASE: {result['only_in_1']}")
        print(f"    Only in vesin: {result['only_in_2']}")
    
    # Case 3: Non-orthogonal cell with mixed PBC
    print("\n--- Case 3: Non-orthogonal cell with mixed PBC ---")
    atoms = ase.Atoms(
        positions=np.random.rand(10, 3) * 5,
        cell=[
            [8.0, 0.0, 0.0],
            [2.0, 7.0, 0.0],  # Tilted
            [0.0, 0.0, 30.0],
        ],
        pbc=[True, True, False]
    )
    
    i_ase, j_ase, d_ase, S_ase = get_ase_neighbor_list(atoms, cutoff)
    
    atoms_exp = expand_cell_for_vesin(atoms, cutoff=cutoff)
    i_vesin, j_vesin, d_vesin, S_vesin = get_vesin_neighbor_list(atoms_exp, cutoff)
    
    result = compare_neighbor_lists(i_ase, j_ase, d_ase, S_ase,
                                     i_vesin, j_vesin, d_vesin, S_vesin)
    passed = result['edges_match']
    all_passed = all_passed and passed
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: ASE={result['n_edges_1']}, vesin={result['n_edges_2']}")
    
    return all_passed


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" VESIN vs ASE MIXED PBC COMPARISON TESTS")
    print("=" * 70)
    
    # Print versions
    print(f"\nVersions: vesin={vesin.__version__}, ase={ase.__version__}")
    print("Note: vesin docs claim mixed PBC is unsupported, but TEST 2 will verify.\n")
    
    results = {}
    
    # Test 1: Comparisons
    results["2d_slab"] = test_2d_slab_comparison()
    results["1d_chain"] = test_1d_chain_comparison()
    results["various_mixed"] = test_various_mixed_pbc()
    
    # Test 2: Direct mixed PBC to vesin
    test_vesin_direct_mixed_pbc()
    
    # Test 3: Edge cases
    results["edge_cases"] = test_edge_cases()
    
    # Summary
    print_separator("SUMMARY")
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    if all_passed:
        print("All tests passed! The large-cell workaround correctly emulates mixed PBC.")
    else:
        print("Some tests failed. Review the output above for details.")
    
    print()
