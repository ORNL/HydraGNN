import os
import torch
import pickle, csv
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from ogb_utils import *

##################################################################################################################

var_names = ["GAP"]
datafile = os.path.join(os.path.dirname(__file__), "dataset/pcqm4m_gap.csv")
smiles_sets, values_sets = datasets_load(datafile)
############################################################
train_tensor = values_sets[0]

print(torch.max(train_tensor, 0)[0])
print(torch.min(train_tensor, 0)[0])
print(torch.mean(train_tensor, 0))
print(torch.std(train_tensor, 0))

outputfile_name = "statistics.pkl"
with open(
    os.path.join(os.path.dirname(__file__), "dataset/" + outputfile_name), "wb"
) as f:
    pickle.dump(torch.max(train_tensor, 0)[0].numpy(), f)
    pickle.dump(torch.min(train_tensor, 0)[0].numpy(), f)
    pickle.dump(torch.mean(train_tensor, 0).numpy(), f)
    pickle.dump(torch.std(train_tensor, 0).numpy(), f)

##################################################################################################################
fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
plt.subplots_adjust(
    left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.35
)
for idata, dataset, datasetname in zip(
    range(3),
    values_sets,
    ["train", "val", "test"],
):
    for ifeat in range(train_tensor.shape[1]):
        ax = axs[idata]
        xfeat = dataset[:, ifeat].squeeze().detach().numpy()
        hist1d, bine, _ = ax.hist(xfeat, bins=50)

        if ifeat == 0:
            ax.set_title(datasetname + ", " + str(len(xfeat)))
        ax.set_xlabel(var_names[ifeat])

fig.savefig(os.path.join(os.path.dirname(__file__), "dataset/train_val_test_hist.png"))
plt.close()
