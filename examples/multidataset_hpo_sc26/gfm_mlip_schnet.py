import os
import sys
import runpy

DEFAULT_MPNN_TYPE = "SchNet"

def main():
    if not any(arg.startswith("--mpnn_type") for arg in sys.argv):
        sys.argv.append(f"--mpnn_type={DEFAULT_MPNN_TYPE}")
    target = os.path.join(os.path.dirname(__file__), "gfm_mlip_all_mpnn.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
