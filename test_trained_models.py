import torch
import os
from models.PNNStack import PNNStack
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader
from data_loading_and_transformation.serialized_dataset_loader import (
    SerializedDataLoader,
)
from data_loading_and_transformation.dataset_descriptors import (
    AtomFeatures,
    Dataset,
)
from data_loading_and_transformation.dataset_descriptors import AtomFeatures
from utilities.utils import load_data, dataset_splitting, test
from utilities.models_setup import generate_model
import tqdm
import json
from utilities.visualizer import Visualizer

models_dir = "./reproducing_results/best_performing_models/"

available_models = os.listdir(models_dir)
for model in available_models:
    print(model)

chosen_model = available_models[
    int(
        input(
            "Type the number of the model for which you want to reproduce the results with(index starts with 0): "
        )
    )
]

with open(f"{models_dir}{chosen_model}/config.json", "r") as f:
    config = json.load(f)

chosen_dataset_option = [x for x in Dataset if x.value == config["dataset_option"]][0]

os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
dataset_CuAu, dataset_FePt, dataset_FeSi = load_data(config)
train_loader, val_loader, test_loader = dataset_splitting(
    dataset_CuAu=dataset_CuAu,
    dataset_FePt=dataset_FePt,
    dataset_FeSi=dataset_FeSi,
    batch_size=config["batch_size"],
    perc_train=config["perc_train"],
    chosen_dataset_option=chosen_dataset_option,
)
model = generate_model(
    model_type="PNN",
    input_dim=len(config["atom_features"]),
    dataset=train_loader.dataset,
    config=config,
)
state_dict = torch.load(
    f"./{models_dir}{chosen_model}/{chosen_model}.pk",
    map_location="cpu",
)
model.load_state_dict(state_dict)

error, true_values, predicted_values, error_values = test(
    test_loader, model, config["output_dim"]
)
print(error)
visualizer = Visualizer("")
visualizer.add_test_values(true_values=true_values, predicted_values=predicted_values)
visualizer.create_scatter_plot(save_plot=False)
