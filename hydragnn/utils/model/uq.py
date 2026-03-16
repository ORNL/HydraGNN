import torch.nn as nn


def enable_dropout_modules(model: nn.Module) -> None:
    model.eval()
    dropout_types = (
        nn.Dropout,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()
import torch


def enable_dropout_modules(model):
    """
    Put the full model in eval mode, then re-enable dropout layers for MCD.
    This keeps BatchNorm frozen while allowing stochastic dropout at inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.train()
    return model

