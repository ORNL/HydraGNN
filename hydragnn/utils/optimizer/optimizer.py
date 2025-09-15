import torch
from hydragnn.utils.distributed import get_device_name
from torch.distributed.optim import ZeroRedundancyOptimizer

deepspeed_available = True
try:
    import deepspeed
except:
    deepspeed_available = False


def select_standard_optimizer(model, config):
    optimizer = None

    if config["type"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    elif config["type"] == "FusedLAMB":
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


def select_zero_redundancy_optimizer(model, config):
    optimizer = None

    if config["type"] == "SGD":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.SGD,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adam":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adadelta":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adadelta,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adagrad":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adagrad,
            lr=config["learning_rate"],
        )
    elif config["type"] == "Adamax":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adamax,
            lr=config["learning_rate"],
        )
    elif config["type"] == "AdamW":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=config["learning_rate"],
        )
    elif config["type"] == "RMSprop":
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.RMSprop,
            lr=config["learning_rate"],
        )
    elif config["type"] == "FusedLAMB":
        assert deepspeed_available, "deepspeed package not installed"
        assert (
            "cpu" != get_device_name()
        ), "GPUs not available to use FusedLAMB optimizer from deepspeed package"
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=deepspeed.ops.lamb.FusedLamb,
            lr=config["learning_rate"],
        )
    else:
        raise NameError("The string used to identify the optimizer is NOT recognized")

    return optimizer


def select_optimizer(model, config):
    use_zero = False

    if "use_zero_redundancy" in config:
        use_zero = config["use_zero_redundancy"]

    if use_zero:
        return select_zero_redundancy_optimizer(model, config)
    else:
        return select_standard_optimizer(model, config)
