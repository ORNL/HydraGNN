import torch


def selected_optimizer(model, config):
    optimizer = None

    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adadelta":
        optimizer = torch.optim.Adamdelta(
            model.parameters(), lr=config["learning_rate"]
        )
    elif config["optimizer"] == "Adagrad":
        optimizer = torch.optim.Adamgrad(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "SparseAdam":
        optimizer = torch.optim.SparseAdam(
            model.parameters(), lr=config["learning_rate"]
        )
    elif config["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSProp(model.parameters(), lr=config["learning_rate"])
    else:
        raise NameError("The string used to identify the optimizer is NOT recognized")

    return optimizer
