from torch_geometric.data import DataLoader
import torch


def train_validate_test(
    model,
    optimizer,
    num_epoch,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
):
    for epoch in range(1, num_epoch):
        loss = train(train_loader, model, optimizer)
        writer.add_scalar("train error", loss, epoch)
        val_mse = test(val_loader, model)
        writer.add_scalar("validate error", val_mse, epoch)
        test_mse = test(test_loader, model)
        writer.add_scalar("test error", test_mse, epoch)
        scheduler.step(val_mse)
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mse:.4f}, "
            f"Test: {test_mse:.4f}"
        )


def train(loader, model, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.train()
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        pred = model(data)
        real_value = torch.reshape(data.y, (data.y.size()[0], 1))
        loss = model.loss(pred, real_value)
        loss.backward()
        total_error += loss.item() * data.num_graphs
        opt.step()
    return total_error / len(loader.dataset)


@torch.no_grad()
def test(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_error = 0
    model.eval()
    for data in loader:
        data = data.to(device)
        pred = model(data)
        real_value = torch.reshape(data.y, (data.y.size()[0], 1))
        error = model.loss(pred, real_value)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset)


def split_dataset(dataset: [], batch_size: int, perc_train: float, perc_val: float):
    data_size = len(dataset)
    train_loader = DataLoader(
        dataset[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset[int(data_size * perc_train) : int(data_size * (perc_train + perc_val))],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset[int(data_size * (perc_train + perc_val)) :],
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader


def combine_and_split_datasets(
    dataset1: [], dataset2: [], batch_size: int, perc_train: float
):
    data_size = len(dataset1)
    train_loader = DataLoader(
        dataset1[: int(data_size * perc_train)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset1[int(data_size * perc_train) :],
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
