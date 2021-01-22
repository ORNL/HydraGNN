import torch
import os
from models.PNNStack import PNNStack
from data_loading_and_transformation.raw_dataset_loader import RawDataLoader
from data_loading_and_transformation.serialized_dataset_loader import SerializedDataLoader
from data_loading_and_transformation.dataset_descriptors import AtomFeatures
from utilities.utils import load_data, dataset_splitting, test
from utilities.models_setup import generate_model
import tqdm

config = {
    "radius": 7,
    "max_num_node_neighbours": 5,
    "predicted_value_option": 5,
    "atom_features": [AtomFeatures.NUM_OF_PROTONS, AtomFeatures.CHARGE_DENSITY],
    "output_dim": 33,
    "batch_size": 64,
    "hidden_dim": 15,
    "num_conv_layers": 16,
    "learning_rate": 0.02,
    "num_epoch": 200,
    "perc_train": 0.7,
    "dataset_option": 'FePt',
}
os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
dataset_CuAu, dataset_FePt = load_data(config)
train_loader, val_loader, test_loader = dataset_splitting(
    dataset1=dataset_CuAu,
    dataset2=dataset_FePt,
    batch_size=config["batch_size"],
    perc_train=config["perc_train"],
    chosen_dataset_option=2
)
model = generate_model(
    model_type="PNN",
    input_dim=2,
    dataset=train_loader.dataset,
    config=config,
)
state_dict = torch.load('./models_serialized/PNNStack-r-7-mnnn-5-ncl-16-hd-15-ne-200-lr-0.02-bs-64-data-FePt-node_ft-2-pred_val-5.pk', map_location='cpu')
model.load_state_dict(state_dict)

result = test(test_loader, model, config["output_dim"])
