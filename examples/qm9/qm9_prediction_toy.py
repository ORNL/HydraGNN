import os, json
import matplotlib.pyplot as plt
from qm9_utils import *
import numpy as np

##################################################################################################################
graph_feature_names = ["HOMO (eV)", "LUMO (eV)", "GAP (eV)"]
dirpwd = os.path.dirname(__file__)
datafile_cut = os.path.join(dirpwd, "dataset/gdb9_gap_cut.csv")
smileslib, gaplib = datasets_load_gap(datafile_cut)
##################################################################################################################
##load trained model directory
log_name = "qm9_gap_eV_fullx"
input_filename = os.path.join("./logs/" + log_name, "config.json")
with open(input_filename, "r") as f:
    config = json.load(f)
world_size, world_rank = hydragnn.utils.setup_ddp()
model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"]["Architecture"],
    verbosity=1,
)
hydragnn.utils.load_existing_model(model, log_name, path="./logs/")
model.eval()
##################################################################################################################
# gdb_5700,c1nc2c([nH]1)CCO2,-0.1883,0.0258,0.2141
smilestr = "c1nc2c([nH]1)CCO2"
smilestr = "CCCC"
for smilestr in ["C" * i for i in range(1, 9)]:
    gap_true = gaplib[smileslib.index(smilestr)]
    gap_pred = gapfromsmiles(smilestr, model)
    print("For ", smilestr, "gap (eV), true = ", gap_true, ", predicted = ", gap_pred)
##################################################################################################################
