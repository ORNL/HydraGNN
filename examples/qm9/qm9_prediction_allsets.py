import os, json
import matplotlib.pyplot as plt
from qm9_utils import *
import numpy as np

##################################################################################################################
graph_feature_names = ["HOMO", "LUMO", "GAP"]
dirpwd = os.path.dirname(__file__)
datafile_cut = os.path.join(dirpwd, "dataset/gdb9_gap_cut.csv")
trainvaltest_splitlists = os.path.join(
    dirpwd, "dataset/qm9_train_test_val_idx_lists.pkl"
)
trainset_statistics = os.path.join(dirpwd, "dataset/qm9_statistics.pkl")

##################################################################################################################
##trained model directory
log_name = "qm9_gap_eV_fullx"
input_filename = os.path.join("./logs/" + log_name, "config.json")
# Configurable run choices (JSON file that accompanies this example script).
with open(input_filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
smiles_sets, values_sets, idxs_sets = datasets_load(
    datafile_cut, trainvaltest_splitlists
)
dataset_lists = [[] for dataset in values_sets]
for idataset, (smileset, valueset, idxset) in enumerate(
    zip(smiles_sets, values_sets, idxs_sets)
):
    for smilestr, ytarget, idx in zip(smileset, valueset, idxset):
        dataset_lists[idataset].append(
            generate_graphdata(idx, smilestr, ytarget, var_config)
        )
trainset = dataset_lists[0]
valset = dataset_lists[1]
testset = dataset_lists[2]

(
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
) = hydragnn.preprocess.create_dataloaders(
    trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
)

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"]["Architecture"],
    verbosity=verbosity,
)
hydragnn.utils.load_existing_model(model, log_name, path="./logs/")
model.eval()
##################################################################################################################

smileslib, gaplib = datasets_load_gap(datafile_cut)
cgaptrue = []
cgappred = []
for smilestr in ["C" * i for i in range(1, 9)]:
    gap_true = gaplib[smileslib.index(smilestr)]
    gap_pred = gapfromsmiles(smilestr, model)
    print("For ", smilestr, "gap (eV), true = ", gap_true, ", predicted = ", gap_pred)
    cgaptrue.append(gap_true)
    cgappred.append(gap_pred)
##################################################################################################################
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
isub = -1
for loader, setname in zip(
    [train_loader, val_loader, test_loader], ["train", "val", "test"]
):
    error, rmse_task, true_values, predicted_values = hydragnn.train.test(
        loader, model, verbosity
    )
    ihead = 0
    head_true = np.asarray(true_values[ihead]).squeeze()
    head_pred = np.asarray(predicted_values[ihead]).squeeze()
    ifeat = var_config["output_index"][ihead]
    outtype = var_config["type"][ihead]
    varname = graph_feature_names[ifeat]

    isub += 1
    ax = axs[isub]
    error_mae = np.mean(np.abs(head_pred - head_true))
    error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
    print(varname, ": ev, mae=", error_mae, ", rmse= ", error_rmse)
    print(rmse_task[ihead])
    print(head_pred.shape, head_true.shape)

    ax.scatter(
        head_true, head_pred, s=7, linewidth=0.5, edgecolor="b", facecolor="none"
    )
    ax.scatter(cgaptrue, cgappred, s=14, linewidth=1.0, color="m", marker="+")
    minv = np.minimum(np.amin(head_pred), np.amin(head_true))
    maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
    ax.plot([minv, maxv], [minv, maxv], "r--")
    ax.set_title(setname + "; " + varname + " (eV)", fontsize=16)
    ax.text(
        minv + 0.1 * (maxv - minv),
        maxv - 0.1 * (maxv - minv),
        "MAE: {:.2f}".format(error_mae),
    )
fig.savefig("./logs/" + log_name + "/" + varname + "_all_C.png")
plt.close()
