#!/usr/bin/env python3

import time
import torch
import sys
import os

# Set up the environment
sys.path.insert(0, '/Users/7ml/Documents/Codes/HydraGNN')
os.environ["PYTHONPATH"] = "/Users/7ml/Documents/Codes/HydraGNN"

def profile_dimenet_vs_others():
    """
    Profile the time taken by different MPNN + EquiformerV2 combinations.
    """
    
    # Simple timer function
    def time_example(mpnn_type, example="qm9"):
        cmd = [
            sys.executable,
            f"examples/{example}/{example}.py",
            "--mpnn_type", mpnn_type,
            "--global_attn_engine", "EquiformerV2",
            "--num_epoch", "1"  # Just 1 epoch for timing
        ]
        
        print(f"Testing {mpnn_type} with EquiformerV2...")
        start_time = time.time()
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"{mpnn_type}: {elapsed:.2f} seconds")
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
        
        return elapsed
    
    # Test different MPNN types
    mpnn_types = ["SAGE", "GIN", "GAT", "SchNet", "DimeNet", "EGNN"]
    
    results = {}
    for mpnn_type in mpnn_types:
        try:
            elapsed = time_example(mpnn_type)
            results[mpnn_type] = elapsed
        except Exception as e:
            print(f"Failed to test {mpnn_type}: {e}")
            results[mpnn_type] = float('inf')
    
    # Sort by time
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print("\n=== TIMING RESULTS ===")
    for mpnn_type, elapsed in sorted_results:
        if elapsed == float('inf'):
            print(f"{mpnn_type}: FAILED")
        else:
            print(f"{mpnn_type}: {elapsed:.2f}s")
    
    # Find the slowest
    slowest = max(results.items(), key=lambda x: x[1] if x[1] != float('inf') else 0)
    fastest = min(results.items(), key=lambda x: x[1])
    
    if slowest[1] != float('inf') and fastest[1] != float('inf'):
        speedup = slowest[1] / fastest[1]
        print(f"\nSlowest: {slowest[0]} ({slowest[1]:.2f}s)")
        print(f"Fastest: {fastest[0]} ({fastest[1]:.2f}s)")
        print(f"Speedup factor: {speedup:.1f}x")

if __name__ == "__main__":
    profile_dimenet_vs_others()