import os, json
import matplotlib.pyplot as plt
from qm9_utils import *

graph_feature_names = ["HOMO(eV)", "LUMO(eV)", "GAP(eV)"]
dirpwd = os.path.dirname(__file__)
datafile_cut = os.path.join(dirpwd, "dataset/gdb9_gap_cut.csv")
trainvaltest_splitlists = os.path.join(
    dirpwd, "dataset/qm9_train_test_val_idx_lists.pkl"
)
trainset_statistics = os.path.join(dirpwd, "dataset/qm9_statistics.pkl")
##################################################################################################################
inputfilesubstr = sys.argv[1]
input_filename = os.path.join(dirpwd, "qm9_" + inputfilesubstr + ".json")
##################################################################################################################
# Configurable run choices (JSON file that accompanies this example script).
with open(input_filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]
var_config["output_names"] = [
    graph_feature_names[item] for ihead, item in enumerate(var_config["output_index"])
]
var_config["input_node_feature_names"] = node_attribute_names
ymax_feature, ymin_feature, ymean_feature, ystd_feature = get_trainset_stat(
    trainset_statistics
)
var_config["ymean"] = ymean_feature.tolist()
var_config["ystd"] = ystd_feature.tolist()
##################################################################################################################
# Always initialize for multi-rank training.
world_size, world_rank = hydragnn.utils.setup_ddp()
##################################################################################################################
norm_yflag = False  # True
smiles_sets, values_sets, idxs_sets = datasets_load(
    datafile_cut, trainvaltest_splitlists
)
dataset_lists = [[] for dataset in values_sets]
for idataset, (smileset, valueset, idxset) in enumerate(
    zip(smiles_sets, values_sets, idxs_sets)
):
    if norm_yflag:
        valueset = (valueset - torch.tensor(var_config["ymean"])) / torch.tensor(
            var_config["ystd"]
        )
        print(valueset[:, 0].mean(), valueset[:, 0].std())
        print(valueset[:, 1].mean(), valueset[:, 1].std())
        print(valueset[:, 2].mean(), valueset[:, 2].std())
    for smilestr, ytarget, idx in zip(smileset, valueset, idxset):
        dataset_lists[idataset].append(generate_graphdata(idx, smilestr, ytarget))
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


config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"]["Architecture"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
)

log_name = "qm9_" + inputfilesubstr + "_eV_fullx"
hydragnn.utils.setup_log(log_name)
writer = hydragnn.utils.get_summary_writer(log_name)
with open("./logs/" + log_name + "/config.json", "w") as f:
    json.dump(config, f)
##################################################################################################################

hydragnn.train.train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    sampler_list,
    writer,
    scheduler,
    config["NeuralNetwork"],
    log_name,
    verbosity,
)

hydragnn.utils.save_model(model, log_name)
hydragnn.utils.print_timers(verbosity)
##################################################################################################################
for ifeat in range(len(var_config["output_index"])):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
    )
    ax = axs[0]
    ax.scatter(
        [trainset[i].cpu().idx for i in range(len(trainset))],
        [trainset[i].cpu().y[ifeat] for i in range(len(trainset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("train, " + str(len(trainset)))
    ax = axs[1]
    ax.scatter(
        [valset[i].cpu().idx for i in range(len(valset))],
        [valset[i].cpu().y[ifeat] for i in range(len(valset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("validate, " + str(len(valset)))
    ax = axs[2]
    ax.scatter(
        [testset[i].cpu().idx for i in range(len(testset))],
        [testset[i].cpu().y[ifeat] for i in range(len(testset))],
        edgecolor="b",
        facecolor="none",
    )
    ax.set_title("test, " + str(len(testset)))
    fig.savefig(
        "./logs/"
        + log_name
        + "/qm9_train_val_test_"
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
        + "/qm9_train_val_test_"
        + var_config["input_node_feature_names"][ifeat]
        + ".png"
    )
    plt.close()
