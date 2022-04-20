import torch
from .distributed import get_device_name

deepspeed_available = True
try:
    import deepspeed
except ImportError:
    deepspeed_available = False


def select_optimizer(model, config):
    optimizer = None

    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "FusedLAMB":
        assert deepspeed_available, "deepspeed package not installed"
        assert (
            "cpu" != get_device_name()
        ), "GPUs not available to use FusedLAMB optimizer from deepspeed package"
        optimizer = deepspeed.ops.lamb.FusedLamb(
            model.parameters(), lr=config["learning_rate"]
        )
    else:
        raise NameError("The string used to identify the optimizer is NOT recognized")

    return optimizer
