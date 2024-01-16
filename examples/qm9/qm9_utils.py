import os, json
import torch
import torch_geometric
import hydragnn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter


def plot_node_graph_features(var_config, train, val, test, output_dir):
    for ifeat in range(len(var_config["output_index"])):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        plt.subplots_adjust(
            left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
        )
        ax = axs[0]
        ax.scatter(
            [train[i].cpu().idx.item() for i in range(len(train))],
            [train[i].cpu().y[ifeat, 0].item() for i in range(len(train))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("train, " + str(len(train)))
        ax = axs[1]
        ax.scatter(
            [val[i].cpu().idx.item() for i in range(len(val))],
            [val[i].cpu().y[ifeat, 0].item() for i in range(len(val))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("validate, " + str(len(val)))
        ax = axs[2]
        ax.scatter(
            [test[i].cpu().idx.item() for i in range(len(test))],
            [test[i].cpu().y[ifeat, 0].item() for i in range(len(test))],
            edgecolor="b",
            facecolor="none",
        )
        ax.set_title("test, " + str(len(test)))
        fig.savefig(
            output_dir
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
            [item for i in range(len(train)) for item in train[i].x[:, ifeat].tolist()],
            "bo",
        )
        ax.set_title("train, " + str(len(train)))
        ax = axs[1]
        ax.plot(
            [item for i in range(len(val)) for item in val[i].x[:, ifeat].tolist()],
            "bo",
        )
        ax.set_title("validate, " + str(len(val)))
        ax = axs[2]
        ax.plot(
            [item for i in range(len(test)) for item in test[i].x[:, ifeat].tolist()],
            "bo",
        )
        ax.set_title("test, " + str(len(test)))
        fig.savefig(
            output_dir
            + "/qm9_train_val_test_"
            + var_config["input_node_feature_names"][ifeat]
            + ".png"
        )
        plt.close()


def plot_predictions_all20(
    model,
    var_config,
    output_dir,
    min_output_feature,
    max_output_feature,
    data_loader,
    filename=None,
):
    ##################################################################################################################
    def scale_back_node(data_arr, minval, maxval, num_nodes):
        minval = minval.to(data_arr.device)
        maxval = maxval.to(data_arr.device)
        data_unit = data_arr * (maxval - minval) + minval
        data_unit *= num_nodes.to(data_arr.device)[:,None]
        return data_unit

    def scale_back(data_arr, minval, maxval):
        minval = minval.to(data_arr.device)
        maxval = maxval.to(data_arr.device)
        data_unit = data_arr * (maxval - minval) + minval
        return data_unit

    _, _, true_values, predicted_values = hydragnn.train.test(data_loader, model, 1)

    num_nodes = torch.zeros(len(data_loader.dataset))
    for idata in range(len(data_loader.dataset)):
        num_nodes[idata] = data_loader.dataset[idata].x.size(0)

    model = model.module
    fig, axs = plt.subplots(4, 5, figsize=(18, 15))
    axs = axs.flatten()
    mae_list = []
    rmse_list = []
    name_list = []
    unit_list = []
    for ihead in range(model.num_heads):
        head_true = true_values[ihead]
        head_pred = predicted_values[ihead]
        ax = axs[ihead]
        unit = var_config["output_units"][ihead]
        varname = var_config["output_names"][ihead]
        if unit == "eV":
            head_true = scale_back_node(
                head_true,
                min_output_feature[ihead],
                max_output_feature[ihead],
                num_nodes,
            )
            head_pred = scale_back_node(
                head_pred,
                min_output_feature[ihead],
                max_output_feature[ihead],
                num_nodes,
            )
        else:
            head_true = scale_back(
                head_true,
                min_output_feature[ihead],
                max_output_feature[ihead],
            )
            head_pred = scale_back(
                head_pred,
                min_output_feature[ihead],
                max_output_feature[ihead],
            )
        error_mae = torch.abs(head_pred - head_true).mean()
        error_rmse = ((head_pred - head_true)**2).mean()**0.5
        print(varname, ": ", unit, ", mae=", error_mae, ", rmse= ", error_rmse)
        print(head_pred.shape, head_true.shape)
        mae_list.append(error_mae)
        rmse_list.append(error_rmse)
        unit_list.append(unit)
        name_list.append(varname)
        head_true = head_true.cpu().numpy()
        head_pred = head_pred.cpu().numpy()
        ax.scatter(
            head_true, head_pred, s=7, linewidth=0.5, edgecolor="b", facecolor="none"
        )
        minv = np.minimum(np.amin(head_pred), np.amin(head_true))
        maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
        ax.plot([minv, maxv], [minv, maxv], "r--")
        if varname == "chargedensity":
            varname = "Partial Charge"
        ax.set_title(varname + "($" + unit + "$)", fontsize=16)
        ax.text(
            minv + 0.1 * (maxv - minv),
            maxv - 0.1 * (maxv - minv),
            "R2: {:.2f}".format(r2_score(head_true, head_pred)),
        )
        yfmt = ScalarFormatter()
        yfmt.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(yfmt)
        ax.xaxis.set_major_formatter(yfmt)
        if ihead in [0, 5, 10, 15]:
            ax.set_ylabel("Predicted")
        if ihead >= 15:
            ax.set_xlabel("True")
    for iext in range(model.num_heads, axs.size):
        axs[iext].axis("off")
    fig.subplots_adjust(wspace=0.25, hspace=0.28)
    if filename is None:
        plt.savefig(output_dir + "/scatterplot.png", bbox_inches="tight")
    else:
        plt.savefig(output_dir + "/%s.png" % filename, bbox_inches="tight")
    plt.close()
