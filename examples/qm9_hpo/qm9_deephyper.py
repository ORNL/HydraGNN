import os, sys, json
import logging

import torch
import torch_geometric

torch.backends.cudnn.enabled = False

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn


# Update each sample prior to loading.
def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    data.y = data.y[:, 10] / len(data.x)
    graph_features_dim = [1]
    node_feature_dim = [1]
    return data


def run(trial):

    global config, log_name, train_loader, val_loader, test_loader

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]

    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()

    log_name = log_name + "_" + str(trial.id)
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    # log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    # Update the config dictionary with the suggested hyperparameters
    config["NeuralNetwork"]["Architecture"]["model_type"] = trial.parameters[
        "model_type"
    ]
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = trial.parameters[
        "hidden_dim"
    ]
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = trial.parameters[
        "num_conv_layers"
    ]

    dim_headlayers = [
        trial.parameters["dim_headlayers"]
        for i in range(trial.parameters["num_headlayers"])
    ]

    for head_type in config["NeuralNetwork"]["Architecture"]["output_heads"]:
        config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "num_headlayers"
        ] = trial.parameters["num_headlayers"]
        config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "dim_headlayers"
        ] = dim_headlayers

    if trial.parameters["model_type"] not in ["EGNN", "SchNet", "DimeNet"]:
        config["NeuralNetwork"]["Architecture"]["equivariance"] = False

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

    hydragnn.utils.save_config(config, log_name)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    # Return the metric to minimize (e.g., validation loss)
    validation_loss, tasks_loss = hydragnn.train.validate(
        val_loader, model, verbosity, reduce_ranks=True
    )

    # Move validation_loss to the CPU and convert to NumPy object
    validation_loss = validation_loss.cpu().detach().numpy()

    # Return the metric to minimize (e.g., validation loss)
    # By default, DeepHyper maximized the objective function, so we need to flip the sign of the validation loss function
    print("validation_loss.item()", validation_loss.item())
    return -validation_loss.item()


if __name__ == "__main__":

    log_name = "qm9"

    # Use built-in torch_geometric dataset.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    dataset = torch_geometric.datasets.QM9(
        root="dataset/qm9", pre_transform=qm9_pre_transform
    )

    trainset, valset, testset = hydragnn.preprocess.split_dataset(dataset, 0.8, False)
    (train_loader, val_loader, test_loader) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, 64
    )

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from deephyper.evaluator import Evaluator

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    problem.add_hyperparameter((1, 2), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((50, 52), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((1, 3), "dim_headlayers")  # discrete parameter
    problem.add_hyperparameter(
        ["EGNN", "PNA", "SchNet", "DimeNet"], "model_type"
    )  # categorical parameter

    # Define the search space for hyperparameters
    # define the evaluator to distribute the computation
    parallel_evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 1,
        },
    )

    # define your search and execute it
    search = CBO(problem, parallel_evaluator, random_state=42, log_dir=log_name)

    timeout = 1200
    results = search.search(max_evals=10, timeout=timeout)
    print(results)

    sys.exit(0)
