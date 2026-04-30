"""
combine_adios.py

Reads steps from structures-0.bp ... structures-7.bp and writes
them all sequentially into a single structures-all.bp.

Usage:
    python combine_adios.py
"""

import sys, glob, socket, getpass
import numpy as np
import adios2.bindings as adios2
from tqdm import tqdm


local_cache_dir = f"/tmp/{getpass.getuser()}"
if len(sys.argv) == 2:
    local_cache_dir = sys.argv[1]


INPUT_FILES = glob.glob(f"{local_cache_dir}/inference_fused_results_gpu*.bp")
OUTPUT_FILE = f"{local_cache_dir}/inference_fused_results_all-{socket.gethostname()}.bp"

VARS_1D_INT = ["atom_types"]
VARS_1D_FLT = [
    "coordinates_x",
    "coordinates_y",
    "coordinates_z",
    "forces_x",
    "forces_y",
    "forces_z",
]
VARS_SCALAR = ["formation_energy"]
ALL_VARS = VARS_1D_INT + VARS_1D_FLT + VARS_SCALAR


def define_output_variables(io):
    """Define all output variables once with dummy shape [1]."""
    dummy_flt = np.zeros(1, dtype=np.float64)
    dummy_int = np.zeros(1, dtype=np.int32)
    vars = {}
    for name in VARS_1D_INT:
        vars[name] = io.DefineVariable(name, dummy_int, [1], [0], [1])
    for name in VARS_1D_FLT + VARS_SCALAR:
        vars[name] = io.DefineVariable(name, dummy_flt, [1], [0], [1])
    return vars


def copy_step(reader, read_io, writer, write_vars):
    """Read one step from reader and write it to writer."""
    N = read_io.InquireVariable("atom_types").Shape()[0]

    # Allocate read buffers
    atom_types = np.zeros(N, dtype=np.int32)
    coords_x = np.zeros(N, dtype=np.float64)
    coords_y = np.zeros(N, dtype=np.float64)
    coords_z = np.zeros(N, dtype=np.float64)
    forces_x = np.zeros(N, dtype=np.float64)
    forces_y = np.zeros(N, dtype=np.float64)
    forces_z = np.zeros(N, dtype=np.float64)
    energy = np.zeros(1, dtype=np.float64)

    # Read
    reader.Get(read_io.InquireVariable("atom_types"), atom_types)
    reader.Get(read_io.InquireVariable("coordinates_x"), coords_x)
    reader.Get(read_io.InquireVariable("coordinates_y"), coords_y)
    reader.Get(read_io.InquireVariable("coordinates_z"), coords_z)
    reader.Get(read_io.InquireVariable("forces_x"), forces_x)
    reader.Get(read_io.InquireVariable("forces_y"), forces_y)
    reader.Get(read_io.InquireVariable("forces_z"), forces_z)
    reader.Get(read_io.InquireVariable("formation_energy"), energy)
    reader.EndStep()  # triggers deferred Gets

    # Resize output variables to current N and write
    for name in VARS_1D_INT + VARS_1D_FLT:
        write_vars[name].SetShape([N])
        write_vars[name].SetSelection([[0], [N]])

    writer.BeginStep()
    writer.Put(write_vars["atom_types"], atom_types)
    writer.Put(write_vars["coordinates_x"], coords_x)
    writer.Put(write_vars["coordinates_y"], coords_y)
    writer.Put(write_vars["coordinates_z"], coords_z)
    writer.Put(write_vars["forces_x"], forces_x)
    writer.Put(write_vars["forces_y"], forces_y)
    writer.Put(write_vars["forces_z"], forces_z)
    writer.Put(write_vars["formation_energy"], energy)
    writer.EndStep()


def combine(input_files, output_file):
    # --- Set up writer ---
    out_adios = adios2.ADIOS()
    out_io = out_adios.DeclareIO("writer")
    out_io.SetEngine("BP5")
    writer = out_io.Open(output_file, adios2.Mode.Write)
    write_vars = define_output_variables(out_io)

    total_steps = 0

    # --- Iterate over input files ---
    in_adios = adios2.ADIOS()
    in_io = in_adios.DeclareIO("reader")
    for input_file in tqdm(input_files, desc="8 files"):
        reader = in_io.Open(input_file, adios2.Mode.Read)

        file_steps = 0
        while reader.BeginStep() == adios2.StepStatus.OK:
            copy_step(reader, in_io, writer, write_vars)
            file_steps += 1
            total_steps += 1

        reader.Close()
        print(f"  {input_file}: {file_steps} steps copied")

    writer.Close()
    print(f"\nDone — {total_steps} total steps written to {output_file}")


def verify(output_file):
    """Read back combined file and print a summary."""
    a = adios2.ADIOS()
    io = a.DeclareIO("reader")
    reader = io.Open(output_file, adios2.Mode.Read)

    print(f"\nVerifying {output_file}:")
    step = 0
    while reader.BeginStep() == adios2.StepStatus.OK:
        N = io.InquireVariable("atom_types").Shape()[0]
        energy = np.zeros(1, dtype=np.float64)
        reader.Get(io.InquireVariable("formation_energy"), energy)
        reader.EndStep()
        # print(f"  [{step:3d}] natoms={N}  formation_energy={energy[0]:.3f} eV")
        step += 1

    reader.Close()


if __name__ == "__main__":
    print(f"Combining {len(INPUT_FILES)} files into {OUTPUT_FILE}\n")
    combine(INPUT_FILES, OUTPUT_FILE)
    # verify(OUTPUT_FILE)
