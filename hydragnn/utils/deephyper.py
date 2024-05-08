import os
import subprocess


def master_from_host(host):
    get_master = "ssh " + host + " hostname -I"
    master_addr = subprocess.check_output(get_master, shell=True)
    master_addr = master_addr.decode().split()[0]
    print("master address =", master_addr)
    return master_addr


def read_node_list():
    node_list = os.environ["SLURM_NODELIST"]
    nodes = []
    system = os.getenv("HYDRAGNN_SYSTEM", "frontier")
    if system == "frontier":
        node_subsets = node_list[9:-1].split(",")
        for subset in node_subsets:
            if "-" in subset:
                start, end = subset.split("-")
                start, end = int(start), int(end)
                for i in range(start, end + 1):
                    leading_zeros = "".join(["0"] * (5 - len(str(i))))
                    nodes.append(f"frontier{leading_zeros}{i}")
            else:
                nodes.append(f"frontier{subset}")
        nodes_string = ",".join(nodes)
    elif system == "perlmutter":
        node_subsets = node_list[4:-1].split(",")
        for subset in node_subsets:
            if "-" in subset:
                start, end = subset.split("-")
                start, end = int(start), int(end)
                for i in range(start, end + 1):
                    leading_zeros = "".join(["0"] * (6 - len(str(i))))
                    nodes.append(f"nid{leading_zeros}{i}")
            else:
                nodes.append(f"nid{subset}")
        nodes_string = ",".join(nodes)
    return nodes, nodes_string


def create_ds_config(job_id, mbs, gbs):
    config_file_name = f"ds_config_{job_id}.json"
    with open(config_file_name, "w") as writer:
        writer.write("{\n")
        writer.write('    "fp16": {\n        "enabled": true\n    },\n')
        writer.write(f'    "train_micro_batch_size_per_gpu" : {mbs},\n')
        writer.write(f'    "train_batch_size" : {gbs},\n')

        text = "\n".join(
            [
                '    "optimizer": {',
                '        "type": "Adam",',
                '        "params": {',
                '            "torch_adam": false,',
                '            "adam_w_mode": false,',
                '            "lr": "auto",',
                '            "betas": "auto",',
                '            "eps": "auto",',
                '            "weight_decay": "auto"',
                "        }",
                "    }\n",
            ]
        )

        writer.write(text)
        writer.write("}")

    return config_file_name


def read_job_node_list(job_id, job_nodes=None):
    if job_nodes is None:
        nodes, nodes_string = read_node_list()
        total_nodes = len(nodes)
        start = (job_id * 4) % total_nodes
        end = start + 4

        job_nodes_string = ",".join(nodes[start:end])
        job_master = master_from_host(nodes[start])
    else:
        job_master = master_from_host(job_nodes[0])
        job_nodes_string = ",".join(job_nodes)

    return job_nodes_string, job_master


def create_launch_command(
    prefix, params, job_id, job_nodes=None, DEEPHYPER_LOG_DIR="."
):

    CHECKPOINT_PATH = "checkpoints/gpt2_345m"
    VOCAB_FILE = "gpt2-vocab.json"
    MERGE_FILE = "gpt2-merges.txt"
    DATA_PATH = (
        "/lustre/orion/world-shared/stf218/sajal/mtds/gptdata/gpttext_article_document"
    )
    print("job_id original:", job_id)
    job_id = int(job_id.split(".")[1])
    world_size = 32
    tps = [1, 2, 4, 8]  # factors of num-heads 24 and 32
    pps = [1, 2, 4, 8]  # factors of num-layers 24
    tp = tps[int(params["tp"])]
    pp = pps[int(params["pp"])]
    mbs = int(params["mbs"])
    gbs = int(params["gbs"])

    if tp * pp > world_size:
        pp = world_size // tp

    dp = world_size // (tp * pp)
    num_micro_batches = dp
    gbs = 10 * int(num_micro_batches * mbs)

    print("Parameters (tp, pp, dp, mbs, gbs):", tp, pp, dp, mbs, gbs)
    ds_config_file = create_ds_config(job_id, mbs, gbs)
    GPT_ARGS = " ".join(
        [
            f"--tensor-model-parallel-size {tp}",
            f"--pipeline-model-parallel-size {pp}",
            f"--num-layers 24",
            f"--hidden-size 2064",
            f"--num-attention-heads 24",
            f"--seq-length 2048",
            f"--max-position-embeddings 2048",
            f"--micro-batch-size {mbs}",
            f"--global-batch-size {gbs}",
            f"--lr 0.00015",
            f"--train-iters 10",
            f"--lr-decay-iters 5",
            f"--lr-decay-style cosine",
            f"--vocab-file {VOCAB_FILE}",
            f"--merge-file {MERGE_FILE}",
            f"--lr-warmup-fraction .01",
            f"--fp16",
        ]
    )

    OUTPUT_ARGS = " ".join(
        [
            "--log-interval 10",
            "--eval-iters 10",
            "--eval-interval 10",
            "--save-interval 10",
        ]
    )
    """
         "--save-interval 500",
            "--eval-interval 100",
            "--eval-iters 10"
    """

    DEEPSPEED_ARGS = f"--deepspeed --deepspeed_config {ds_config_file} --deepspeed-activation-checkpointing --checkpoint-activations"
    python_exe = "python"
    python_script = "pretrain_gpt_deepspeed.py"
    print("job_id =", job_id)
    job_nodes, job_master = read_job_node_list(job_id, job_nodes)
    SLURM_ARGS = f"--nodelist {job_nodes} --time=20 --output {DEEPHYPER_LOG_DIR}/output_{job_id}.txt --error {DEEPHYPER_LOG_DIR}/error_{job_id}.txt"
    params = (
        GPT_ARGS
        + " "
        + OUTPUT_ARGS
        + f" --data-path {DATA_PATH}"
        + " "
        + DEEPSPEED_ARGS
        + f" --master-addr {job_master}"
    )

    command = f"{prefix} {SLURM_ARGS} {python_exe} {python_script} {params}"
    # print("Command =", command)
    return command
