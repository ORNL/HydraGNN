import os, sys, json
import logging

import torch
import torch_geometric

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn

import optuna

import pandas as pd

# Update each sample prior to loading.
def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    data.y = data.y[:, 10] / len(data.x)
    graph_features_dim = [1]
    node_feature_dim = [1]
    return data


def objective(trial):

    global config, best_trial_id, best_validation_loss

    # Extract the unique trial ID
    trial_id = trial.number  # or trial_id = trial.trial_id

    log_name = "qm9"
    log_name = log_name + "_" + str(trial_id)
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    # log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    # Define the search space for hyperparameters
    model_type = trial.suggest_categorical("model_type", ["EGNN", "PNA", "SchNet"])
    hidden_dim = trial.suggest_int("hidden_dim", 50, 300)
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
    num_headlayers = trial.suggest_int("num_headlayers", 1, 3)
    dim_headlayers = [
        trial.suggest_int(f"dim_headlayer_{i}", 50, 300) for i in range(num_headlayers)
    ]

    # Update the config dictionary with the suggested hyperparameters
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = hidden_dim
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = num_conv_layers
    # config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["num_headlayers"] = num_headlayers
    # config["NeuralNetwork"]["Architecture"]["output_heads"]["node"]["dim_headlayers"] = dim_headlayers
    config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"][
        "num_headlayers"
    ] = num_headlayers
    config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"][
        "dim_headlayers"
    ] = dim_headlayers

    if model_type not in ["EGNN", "SchNet", "DimeNet"]:
        config["NeuralNetwork"]["Architecture"]["equivariance"] = False

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

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

    """
    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    """

    # Return the metric to minimize (e.g., validation loss)
    validation_loss, tasks_loss = hydragnn.train.validate(
        val_loader, model, verbosity, reduce_ranks=True
    )

    # Move validation_loss to the CPU and convert to NumPy object
    validation_loss = validation_loss.cpu().detach().numpy()

    # Append trial results to the DataFrame
    trial_results.loc[trial_id] = [
        trial_id,
        hidden_dim,
        num_conv_layers,
        num_headlayers,
        dim_headlayers,
        model_type,
        validation_loss,
    ]

    # Update information about the best trial
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_trial_id = trial_id

    # Return the metric to minimize (e.g., validation loss)
    return validation_loss


if __name__ == "__main__":

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()

    log_name = "qm9_test"
    # Enable print to log file.
    hydragnn.utils.setup_log(log_name)

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    dataset = torch_geometric.datasets.QM9(
        root="dataset/qm9", pre_transform=qm9_pre_transform
    )
    trainset, valset, testset = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    sampler = optuna.samplers.TPESampler(consider_prior=True, consider_magic_clip=False)
    # sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.CmaEsSampler(cma_stds=[1.0, 1.0, 1.0], consider_pruned_trials=True, consider_prior=False)
    # sampler = optuna.samplers.GridSampler(consider_prior=False)
    # sampler = optuna.samplers.NSGAIISampler(pop_size=100, crossover_prob=0.9, mutation_prob=0.1)

    # Create an empty DataFrame to store trial results
    trial_results = pd.DataFrame(
        columns=[
            "Trial_ID",
            "Hidden_Dim",
            "Num_Conv_Layers",
            "Num_Headlayers",
            "Dim_Headlayers",
            "Model_Type",
            "Validation_Loss",
        ]
    )

    # Variables to store information about the best trial
    best_trial_id = None
    best_validation_loss = float("inf")

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    # Update the best trial information directly within the DataFrame
    best_trial_info = pd.Series(
        {"Trial_ID": best_trial_id, "Best_Validation_Loss": best_validation_loss}
    )
    trial_results = trial_results.append(best_trial_info, ignore_index=True)

    # Save the trial results to a CSV file
    trial_results.to_csv("hpo_results.csv", index=False)

    # Get the best hyperparameters and corresponding trial ID
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    best_trial_id = (
        study.best_trial.number
    )  # or best_trial_id = study.best_trial.trial_id

    # Print information about the best trial
    print("Best Trial ID:", best_trial_id)
    print("Best Validation Loss:", best_validation_loss)

    sys.exit(0)
