"""
eval_gnn_powerflow.py
Loads the trained model from 'Best_GNN_GNN_NN_model.pt', evaluates on the validation set, and creates parity plots.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import pandapower.networks as nw
import json
from gnn_model import My_GNN_GNN_NN

def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    return dataset[:int(data_size * percentage / 100)]

def make_dataset(dataset, n_bus):
    x_raw, y_raw = [], []
    for i in range(len(dataset)):
        x_sample, y_sample = [], []
        for n in range(n_bus):
            is_pv = 0
            is_pq = 0
            is_slack = 0
            if n == 0:
                is_slack = 1
            elif dataset[i, 4 * n + 2] == 0:
                is_pv = 1
            else:
                is_pq = 1
            x_sample.append([
                dataset[i, 4 * n + 1],
                dataset[i, 4 * n + 2],
                dataset[i, 4 * n + 3],
                dataset[i, 4 * n + 4],
                is_pv,
                is_pq,
                is_slack
            ])
            y_sample.append([
                dataset[i, 4 * n + 3],
                dataset[i, 4 * n + 4]
            ])
        x_raw.append(x_sample)
        y_raw.append(y_sample)
    x_raw = torch.tensor(x_raw, dtype=torch.float)
    y_raw = torch.tensor(y_raw, dtype=torch.float)
    return x_raw, y_raw

def normalize_dataset(x, y):
    x_mean, x_std = torch.mean(x, 0), torch.std(x, 0)
    y_mean, y_std = torch.mean(y, 0), torch.std(y, 0)
    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1
    x_norm = (x - x_mean) / x_std
    x_norm[:, :, 4] = x[:, :, 4]
    y_norm = (y - y_mean) / y_std
    return x_norm, y_norm, x_mean, y_mean, x_std, y_std

def denormalize_output(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean

if __name__ == "__main__":
    bus_system = "118Bus"  # Change as needed
    n_bus = 14
    if bus_system == "14Bus":
        net = nw.case14()
        dataset2 = pd.read_excel("Datasets/14Bus/PF_Dataset_2.xlsx").values
    elif bus_system == "30Bus":
        n_bus = 30
        net = nw.case30()
        dataset2 = pd.read_excel("Datasets/30Bus/PF_Dataset_2_10000.xlsx").values
    elif bus_system == "57Bus":
        n_bus = 57
        net = nw.case57()
        dataset2 = pd.read_excel("Datasets/57Bus/PF_Dataset_2_10000.xlsx").values
    elif bus_system == "118Bus":
        n_bus = 118
        net = nw.case118()
        dataset2 = pd.read_excel("Datasets/118Bus/PF_Dataset_2_10000.xlsx").values
    else:
        raise ValueError("Invalid bus system.")

    val_percentage = 20 if bus_system != "14Bus" else 100
    val_dataset = slice_dataset(dataset2, val_percentage)
    x_raw_val, y_raw_val = make_dataset(val_dataset, n_bus)
    x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std = normalize_dataset(x_raw_val, y_raw_val)
    from_buses = net.line['from_bus'].values
    to_buses = net.line['to_bus'].values
    edge_index = torch.tensor([list(from_buses) + list(to_buses), list(to_buses) + list(from_buses)], dtype=torch.long)
    val_data_list = [Data(x=x, y=y, edge_index=edge_index) for x, y in zip(x_norm_val, y_norm_val)]
    val_loader = DataLoader(val_data_list, batch_size=16)

    # Load hyperparameters
    with open(f"[{n_bus} bus] Model_Hyperparameters.json", "r") as file:
        hyperparams = json.load(file)
    feat_in = hyperparams["Input Features"]
    feat_size1 = hyperparams["GNN Layer 1 Size"]
    feat_size2 = hyperparams["GNN Layer 2 Size"]
    hidden_size1 = hyperparams["Hidden Layer Size (FC)"]
    output_size = hyperparams["Output Size"]
    gnn_type = hyperparams["GNN Type"]
    dropout = hyperparams["Dropout Rate"]
    use_batch_norm = hyperparams["Use Batch Norm"]

    # Load model and weights
    model = My_GNN_GNN_NN(
        node_size=n_bus,
        feat_in=feat_in,
        feat_size1=feat_size1,
        feat_size2=feat_size2,
        hidden_size1=hidden_size1,
        output_size=output_size,
        gnn_type=gnn_type,
        dropout=dropout,
        use_batch_norm=use_batch_norm
    )
    model_filename = f"[{n_bus} bus] Best_GNN_GNN_NN_model.pt"
    model.load_weights(model_filename)
    print(f"Loaded model weights from '{model_filename}'.")

    # Validation
    model.eval()
    y_val_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            y_val_pred = model(batch)
            if y_val_pred.size(0) == 0:
                continue
            y_val_pred = y_val_pred.view(-1, n_bus, 2)
            y_val_predictions.append(y_val_pred)
    y_val_predictions = torch.cat(y_val_predictions, dim=0)
    y_val_targets = torch.cat([batch.y.view(-1, n_bus, 2) for batch in val_loader], dim=0)

    # Denormalize outputs
    y_val_pred_denorm = denormalize_output(y_val_predictions, y_val_mean, y_val_std)
    y_val_targets_denorm = denormalize_output(y_val_targets, y_val_mean, y_val_std)

    # Parity plots for V and delta
    for i, label in enumerate(["Voltage Magnitude (V)", "Voltage Angle (delta)"]):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_val_targets_denorm[:, :, i].flatten(), y_val_pred_denorm[:, :, i].flatten(), alpha=0.5)
        plt.plot([y_val_targets_denorm[:, :, i].min(), y_val_targets_denorm[:, :, i].max()],
                 [y_val_targets_denorm[:, :, i].min(), y_val_targets_denorm[:, :, i].max()],
                 color='red', linestyle='--', label='Ideal')
        plt.xlabel(f'True {label}')
        plt.ylabel(f'Predicted {label}')
        plt.title(f'Parity Plot for {label} (Validation Set)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'parity_plot_{label.replace(" ", "_").lower()}.png', dpi=300)
        plt.close()
        print(f"Saved parity plot for {label}.")
