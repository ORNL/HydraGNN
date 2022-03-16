import os, json
import matplotlib.pyplot as plt
from ogb_utils import *
import numpy as np

graph_feature_names = ["GAP"]
dirpwd = os.path.dirname(__file__)
datafile = os.path.join(dirpwd, "dataset/pcqm4m_gap.csv")
trainset_statistics = os.path.join(dirpwd, "dataset/statistics.pkl")
##################################################################################################################
log_name = "ogb_gap_eV_fullx"
input_filename = os.path.join("./logs/" + log_name, "config.json")
# Configurable run choices (JSON file that accompanies this example script).
with open(input_filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]
##################################################################################################################
# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
norm_yflag = False  # True
smiles_sets, values_sets = datasets_load(datafile)
dataset_lists = [[] for dataset in values_sets]
for idataset, (smileset, valueset) in enumerate(zip(smiles_sets, values_sets)):
    if norm_yflag:
        valueset = (valueset - torch.tensor(var_config["ymean"])) / torch.tensor(
            var_config["ystd"]
        )
        print(valueset[:, 0].mean(), valueset[:, 0].std())
        print(valueset[:, 1].mean(), valueset[:, 1].std())
        print(valueset[:, 2].mean(), valueset[:, 2].std())
    for smilestr, ytarget in zip(smileset, valueset):
        dataset_lists[idataset].append(generate_graphdata(smilestr, ytarget,var_config))
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
    minv = np.minimum(np.amin(head_pred), np.amin(head_true))
    maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
    ax.plot([minv, maxv], [minv, maxv], "r--")
    ax.set_title(setname + "; " + varname + " (eV)", fontsize=16)
    ax.text(
        minv + 0.1 * (maxv - minv),
        maxv - 0.1 * (maxv - minv),
        "MAE: {:.2f}".format(error_mae),
    )
fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
plt.close()
##################################################################################################################
for ifeat in range(len(var_config["output_index"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.scatter(
        range(len(trainset)),
        [trainset[i].cpu().y[ifeat] for i in range(len(trainset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.scatter(
        range(len(valset)),
        [valset[i].cpu().y[ifeat] for i in range(len(valset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.scatter(
        range(len(testset)),
        [testset[i].cpu().y[ifeat] for i in range(len(testset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/ogb_train_val_test_"
        + var_config["output_names"][ifeat]
        + ".png"
    )
    plt.close()

for ifeat in range(len(var_config["input_node_features"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.plot(
        [
            item
            for i in range(len(trainset))
            for item in trainset[i].x[:, ifeat].tolist()
        ],
        "bo",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.plot(
        [item for i in range(len(valset)) for item in valset[i].x[:, ifeat].tolist()],
        "bo",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.plot(
        [item for i in range(len(testset)) for item in testset[i].x[:, ifeat].tolist()],
        "bo",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/ogb_train_val_test_"
        + var_config["input_node_feature_names"][ifeat]
        + ".png"
    )
    plt.close()
