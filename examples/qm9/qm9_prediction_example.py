import os, json
import matplotlib.pyplot as plt
from qm9_utils import *
import numpy as np

##################################################################################################################
graph_feature_names = ["HOMO (eV)", "LUMO (eV)", "GAP (eV)"]
dirpwd = os.path.dirname(__file__)
# gdb_5700,c1nc2c([nH]1)CCO2,-0.1883,0.0258,0.2141
smilestr = "c1nc2c([nH]1)CCO2"
gap_true = 0.2141 * HAR2EV
idx = 5700
##idx and gap_true can be replaced by random numbers when use
data_graph = generate_graphdata(idx, smilestr, gap_true)
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
##################################################################################################################
pred = model(data_graph)
print(
    "For gdb_", idx, "gap (eV), true = ", gap_true, " predicted = ", pred[0][0].item()
)
##################################################################################################################
