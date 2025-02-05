import subprocess
import argparse


def submit_job(nodes, width, depth, dataset_size, zero=False, ckpt=False):
    # Command to execute
    command = [
        "sbatch",
        "-N",
        str(nodes),
        "job-perlmutter.sh",
        str(width),
        str(depth),
        str(dataset_size),
        str(zero),
        str(ckpt),
    ]
    # Run the command and capture output
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    stdout, stderr = process.communicate()
    # Extract the job ID
    output = stdout.strip()
    job_id = int(output.split()[-1])
    return job_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit jobs with varying parameters.")
    parser.add_argument("--width", type=int, required=True, help="Width of the model.")
    parser.add_argument("--depth", type=int, required=True, help="Depth of the model.")

    parser.add_argument(
        "--zero",
        action="store_true",
        help="enable zero optimizer with stage 1",
        default=False,
    )
    parser.add_argument(
        "--ckpt",
        action="store_true",
        help="enable checkpointing for conv layers",
        default=False,
    )

    args = parser.parse_args()

    dataset_size_list = [0.1, 0.2, 0.4, 0.6]
    nodes_list = [8, 16, 32, 32]

    for dataset_size, nodes in zip(dataset_size_list, nodes_list):
        job_id = submit_job(
            nodes, args.width, args.depth, dataset_size, args.zero, args.ckpt
        )
        print(job_id)