"""
train_gnn_powerflow.py
Loads data, trains the GNN model, and saves checkpoints/hyperparameters.
"""
import os
import torch
from torch_geometric.data import Data, DataLoader
from gnn_model import My_GNN_GNN_NN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower.networks as nw
import json

# Utility functions and model class (identical to original)
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

def MSE(yhat, y):
    return torch.mean((yhat - y) ** 2)


if __name__ == "__main__":
    bus_system = "118Bus"  # Change as needed
    n_bus = 14
    if bus_system == "14Bus":
        net = nw.case14()
        dataset1 = pd.read_excel("Datasets/14Bus/PF_Dataset_1.xlsx").values
        dataset2 = pd.read_excel("Datasets/14Bus/PF_Dataset_2.xlsx").values
    elif bus_system == "30Bus":
        n_bus = 30
        net = nw.case30()
        dataset1 = pd.read_excel("Datasets/30Bus/PF_Dataset_1_10000.xlsx").values
        dataset2 = pd.read_excel("Datasets/30Bus/PF_Dataset_2_10000.xlsx").values
    elif bus_system == "57Bus":
        n_bus = 57
        net = nw.case57()
        dataset1 = pd.read_excel("Datasets/57Bus/PF_Dataset_1_10000.xlsx").values
        dataset2 = pd.read_excel("Datasets/57Bus/PF_Dataset_2_10000.xlsx").values
    elif bus_system == "118Bus":
        n_bus = 118
        net = nw.case118()
        dataset1 = pd.read_excel("Datasets/118Bus/PF_Dataset_1_10000.xlsx").values
        dataset2 = pd.read_excel("Datasets/118Bus/PF_Dataset_2_10000.xlsx").values
    else:
        raise ValueError("Invalid bus system.")

    train_percentage = 100
    val_percentage = 20 if bus_system != "14Bus" else 100
    train_dataset = slice_dataset(dataset1, train_percentage)
    val_dataset = slice_dataset(dataset2, val_percentage)
    x_raw_train, y_raw_train = make_dataset(train_dataset, n_bus)
    x_raw_val, y_raw_val = make_dataset(val_dataset, n_bus)
    x_norm_train, y_norm_train, x_train_mean, y_train_mean, x_train_std, y_train_std = normalize_dataset(x_raw_train, y_raw_train)
    x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std = normalize_dataset(x_raw_val, y_raw_val)
    from_buses = net.line['from_bus'].values
    to_buses = net.line['to_bus'].values
    edge_index = torch.tensor([list(from_buses) + list(to_buses), list(to_buses) + list(from_buses)], dtype=torch.long)
    train_data_list = [Data(x=x, y=y, edge_index=edge_index) for x, y in zip(x_norm_train, y_norm_train)]
    val_data_list = [Data(x=x, y=y, edge_index=edge_index) for x, y in zip(x_norm_val, y_norm_val)]
    train_loader = DataLoader(train_data_list, batch_size=16)
    val_loader = DataLoader(val_data_list, batch_size=16)

    # Model hyperparameters
    feat_in = 7
    feat_size1 = 12
    feat_size2 = 12
    hidden_size1 = 128
    output_size = n_bus * 2
    gnn_type = 'GraphConv'  # Change as needed
    dropout = 0
    use_batch_norm = True
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
    learning_rate = 5e-5
    lambda_l2 = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

    # Training loop
    train_loss_list, val_loss_list = [], []
    patience_count = 100
    count = 0
    best_epoch = 0
    lossMin = float('inf')
    for epoch in range(300):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_size = batch.num_graphs
            y_val_mean_expanded = y_val_mean.view(-1).repeat(batch_size)
            y_val_std_expanded = y_val_std.view(-1).repeat(batch_size)
            y_pred = model(batch)
            y_pred = y_pred.view(-1)
            loss = MSE(
                denormalize_output(y_pred, y_val_mean_expanded, y_val_std_expanded),
                denormalize_output(batch.y.view(-1), y_val_mean_expanded, y_val_std_expanded)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_size = batch.num_graphs
                y_val_mean_expanded = y_val_mean.view(-1).repeat(batch_size)
                y_val_std_expanded = y_val_std.view(-1).repeat(batch_size)
                y_val_pred = model(batch)
                y_val_pred = y_val_pred.view(-1)
                loss = MSE(
                    denormalize_output(y_val_pred, y_val_mean_expanded, y_val_std_expanded),
                    denormalize_output(batch.y.view(-1), y_val_mean_expanded, y_val_std_expanded)
                )
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)
        val_loss_list.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < lossMin:
            lossMin = val_loss
            count = 0
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            model_filename = f"[{n_bus} bus] Best_GNN_GNN_NN_model.pt"
            model.save_weights(model_filename)
        else:
            count += 1
            if count > patience_count:
                print(f"Early stopping at epoch {epoch} | Best epoch: {best_epoch}")
                print(f"Best train loss: {best_train_loss:.7f} | Best val loss: {best_val_loss:.7f}")
                break
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train loss: {train_loss:.7f} | Val loss: {val_loss:.7f}")
    print("Training complete.")
    print(f"Best Epoch: {best_epoch} | Best Train Loss: {best_train_loss:.7f} | Best Val Loss: {best_val_loss:.7f}")

    # Save hyperparameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hyperparams = {
        "Number of Buses": n_bus,
        "Learning Rate": learning_rate,
        "L2 Regularization (Lambda)": lambda_l2,
        "Dropout Rate": dropout,
        "Use Batch Norm": use_batch_norm,
        "Input Features": feat_in,
        "GNN Layer 1 Size": feat_size1,
        "GNN Layer 2 Size": feat_size2,
        "Hidden Layer Size (FC)": hidden_size1,
        "Output Size": output_size,
        "GNN Type": gnn_type,
        "Number of GNN Layers": 2,
        "Number of Fully Connected Layers": 2,
        "Total Trainable Parameters": total_params,
    }
    json_filename = f"[{n_bus} bus] Model_Hyperparameters.json"
    with open(json_filename, "w") as file:
        json.dump(hyperparams, file, indent=4)
    print(f"Hyperparameters saved to '{json_filename}'.")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.title(f'Loss Curves for GNN Model Predicting V and δ on IEEE {n_bus}-Bus Dataset', fontsize=14)
    plt.plot(train_loss_list, label="Train Loss", color='blue', linewidth=2)
    plt.plot(val_loss_list, label="Validation Loss", color='orange', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    figure_filename = f"[{n_bus} bus] Loss Curves.png"
    plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved as '{figure_filename}'!")
