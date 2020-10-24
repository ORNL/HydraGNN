from torch_geometric.data import DataLoader
import torch


def train(loader, model, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_error = 0
    model.eval()
    for data in loader:
        data = data.to(device)
        pred = model(data)
        real_value = torch.reshape(data.y, (data.y.size()[0], 1))
        error = model.loss(pred, real_value)
        total_error += error.item() * data.num_graphs

    return total_error / len(loader.dataset)