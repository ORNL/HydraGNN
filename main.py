import torch
from ray import tune
from hyperopt import hp
from ray.tune import CLIReporter
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from functools import partial
import os
from utilities.utils import (
    load_data, 
    combine_and_split_datasets, 
    train_validate_test_hyperopt, 
    train_validate_test_normal 
)
from utilities.models_setup import generate_model
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loading_and_transformation.serialized_dataset_loader import SerializedDataLoader
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader
from torch.utils.tensorboard import SummaryWriter


def run_with_hyperparameter_optimization():
    config = {"batch_size": hp.choice("batch_size", [64]),
            "learning_rate": hp.choice("learning_rate", [0.005]),
            "num_conv_layers": hp.choice("num_conv_layers", [8, 10, 12, 14]),
            "hidden_dim": hp.choice("hidden_dim", [20]),
            "radius": hp.choice("radius", [5, 10, 15, 20, 25]),
            "max_num_node_neighbours": hp.randint("max_num_node_neighbours", [5, 10, 15, 20, 25, 30]),
            }

    algo = HyperOptSearch(space=config, metric="test_mae", mode="min")
    algo = ConcurrencyLimiter(searcher=algo, max_concurrent=10)
    
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='test_mae',
        mode='min',
        grace_period=10)

    reporter = CLIReporter(
        metric_columns=["train_mae", "val_mae", "test_mae", "training_iteration"])

    result = tune.run(
        partial(train_validate_test_hyperopt, checkpoint_dir="./checkpoint-ray-tune"),
        resources_per_trial={"cpu":0.5, "gpu": 0.1},
        search_alg=algo,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter)

def run_normal():
    atom_features = [
        AtomFeatures.NUM_OF_PROTONS,
        AtomFeatures.CHARGE_DENSITY,
        ]
    structure_features = [StructureFeatures.FREE_ENERGY]

    config = {"batch_size": 64, "hidden_dim": 15,}
    config["num_conv_layers"] = int(input("Select number of convolutional layers: "))
    config["learning_rate"] = float(input("Select learning rate: "))
    config["radius"] = int(input("Select the radius within which neighbours of an atom are chosen: "))
    config["max_num_node_neighbours"] = int(input("Select the maximum number of atom neighbours: "))
    
    input_dim = len(atom_features)
    perc_train = 0.7
    dataset1, dataset2 = load_data(config, structure_features, atom_features)

    train_loader, val_loader, test_loader = combine_and_split_datasets(
        dataset1=dataset1, dataset2=dataset2, batch_size=config["batch_size"], perc_train=perc_train
    )

    model_choices = {"1": "GIN", "2": "PNN", "3": "GAT", "4": "MFC"}
    print("Select which model you want to use: 1) GIN 2) PNN 3) GAT 4) MFC")
    chosen_model = model_choices[input("Selected value: ")]
    model = generate_model(model_type=chosen_model, input_dim=input_dim, dataset=dataset1[:int(len(dataset1)*perc_train)], config=config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )
    num_epoch = 200

    model_name = (
        model.__str__()
        + "-r-"
        + str(config["radius"])
        + "-mnnn-"
        + str(config["max_num_node_neighbours"])
        + "-num_conv_layers-"
        + str(model.num_conv_layers)
        + "-hd-"
        + str(model.hidden_dim)
        + "-ne-"
        + str(num_epoch)
        + "-lr-"
        + str(config["learning_rate"])
        + "-bs-"
        + str(config["batch_size"])
        + ".pk"
    )
    writer = SummaryWriter("./logs/" + model_name)
    train_validate_test_normal(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
    )
    torch.save(model.state_dict(), "./models_serialized/" + model_name)


os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
type_of_run = {1: run_with_hyperparameter_optimization, 2: run_normal}

print("Training and validation is conducted on first dataset and testing on the second dataset.")
choice = int(input("Select the type of run between hyperparameter optimization and normal run 1)Hyperopt 2)Normal: "))
type_of_run[choice]()

'''
model.load_state_dict(torch.load("models_serialized/PNNStack7-mnnn-5-hd-75-ne-200-lr-0.01.pk", map_location=torch.device('cpu')))
print(test(test_loader, model))


'''
