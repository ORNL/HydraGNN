# %%
import json
import importlib
import torch
import torch_geometric
import pickle
import os

# deprecated in torch_geometric 2.0q
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

# import hydragnn

import voronoi_utils_edge as voronoi_utils
importlib.reload(voronoi_utils)

import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
# %matplotlib inline

import random

import hydragnn
importlib.reload(hydragnn)

from hydragnn.utils.distributed import get_device

random.seed(42)
np.random.seed(42)
torch.manual_seed(42) 
#torch.use_deterministic_algorithms(True)
dataset = voronoi_utils.VoronoiNet(root='.',force_reload=True) # 

# %%
dataset_dict = vars(dataset)
y_all = torch.stack([data.y for data in dataset_dict['dataset']]).numpy()
y_min = dataset.y_min
y_range = dataset.y_range
y_all_rescaled = y_all*y_range.numpy()+y_min.numpy()

x_all = torch.cat([data.x for data in dataset_dict['dataset']], dim=0).numpy()
x_min = dataset.x_min
x_range = dataset.x_range
x_all_rescaled = x_all*x_range.numpy()+x_min.numpy()

# %%
perm = torch.tensor([319,  12,  81, 137, 351,  34, 103, 258, 322, 118, 307, 252, 284, 293,
        163, 291,  21, 201, 312, 187, 349, 246, 235, 203, 101,  83, 194, 129,
        133, 272, 158, 124, 317, 268,  37, 172,  14, 162, 105, 113, 279, 354,
          8, 393, 170,  52, 396,  86, 374, 225,   6, 372, 238,  44, 224,  19,
        320,  54, 237, 245, 115, 278, 242, 205, 303, 357,  41, 256, 298, 375,
       220,  65, 111,  94, 324, 292, 335, 161,  95, 380, 151, 202, 346, 231,
       149, 325, 333, 119,  42,  99, 106, 207, 355, 330, 184, 281, 146, 232,
         3, 369, 306, 261, 344, 358, 204, 387, 309, 211,  63, 222, 361, 301,
       260, 288, 345,  46,   9, 388, 155,  64, 153, 247, 191,  61,  36, 287,
       255, 141, 316, 348, 125, 399,  70, 311,  53,  38,  15, 304, 365,  32,
        10, 290, 336, 159,  79, 389, 223, 285,  20, 126, 264, 175, 193, 167,
       107, 171, 244, 189, 134,  91, 283, 179, 318, 230,  39,  25, 112,  59,
        11, 152, 249, 366, 228, 371, 182, 381, 394, 132, 385, 343, 271, 254,
        78, 305, 367, 282, 377, 362,  48, 120, 337, 160, 145,  97,  23,  77,
       199, 379,  87, 269, 263, 192, 378,  50, 323, 174, 350,  71,  93, 262,
       383, 259, 123,  85, 390, 135, 314, 164,  28, 190,  26, 294,  45, 173,
       197,   1, 110,  31, 215, 275, 327, 143, 116, 217, 154, 328, 196, 214,
        67, 286,   5,  96, 178, 165, 136, 313, 347,   7,  76, 157, 310, 382,
        74,  75,  90, 195, 156, 329, 338,  49,  68, 121, 176, 359, 188, 186,
       147,  92, 169,  47, 257, 168, 243, 297,  89, 138, 239,   2,  84, 114,
       332, 180, 397, 208,  66, 300, 128,  18, 229, 295, 226, 331,  29, 276,
       227, 251, 392, 117, 360,  55, 221,  17, 148, 363, 206,  40, 209, 341,
       100,  57, 364, 368, 108, 308,  27, 373, 386,  62, 102, 236,  98, 200,
       130, 342, 289, 166, 321,  16, 139, 270, 240, 280, 150, 127,  33,  22,
        82, 233, 198, 241,  73, 299, 250, 326,  88, 266, 109, 273,  56,  35,
        30, 177, 210, 356,  58, 185, 144, 140, 265,  80, 183,  51,   0,   4,
        69, 104, 181,  72, 218, 142, 395, 296, 253,  60, 340, 353, 398, 339,
        24, 219, 277, 315, 352, 370, 376, 391, 274, 234, 131, 248, 334, 216,
       122, 213,  43, 384, 212,  13, 302, 267])
# perm = torch.randperm(len(dataset))
shuffled_dataset = dataset.index_select(perm)
# perm

# %%
try:
    os.environ["SERIALIZED_DATA_PATH"]
except:
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

filename = os.path.join(os.getcwd(), "voronoi_pna_c_vector_small.json")

with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]
var_config = config["NeuralNetwork"]["Variables_of_interest"]

world_size, world_rank = hydragnn.utils.setup_ddp()
log_name = "input_features/all_g5"

hydragnn.utils.setup_log(log_name)

train, val, test = hydragnn.preprocess.split_dataset(
    shuffled_dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
)
(train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
    train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
)

config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

model = hydragnn.models.create_model_config(
    config=config["NeuralNetwork"],
    verbosity=verbosity,
)
model = hydragnn.utils.get_distributed_model(model, verbosity)

learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.0000001
)
writer = hydragnn.utils.get_summary_writer(log_name)
hydragnn.utils.save_config(config, log_name)

hydragnn.train.train_validate_test(
    model,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
    config["NeuralNetwork"],
    log_name,
    verbosity,
    create_plots=config["Visualization"]["create_plots"],
)

# %%
os.mkdir(f'./logs/{log_name}/input_features')
hydragnn.utils.save_model(model, optimizer, log_name)
hydragnn.utils.print_timers(verbosity)

# %%
num_samples = len(test)
true_values = []
predicted_Cij = []
predicted_Nf_strength = []
predicted_Sf_strength = []
predicted_Beta_anisotropy = []
test_MAE = 0.0
variable_index = 3

for data_id, data in enumerate(test):
    predicted = model(data.to(get_device()))
    predicted_C = predicted[variable_index].flatten()
    predicted_Nf = predicted[0].flatten()[0]
    predicted_Sf = predicted[1].flatten()[0]
    predicted_Beta = predicted[2].flatten()[0]
    true = data.y
    predicted_Cij.append(predicted_C.cpu().tolist())
    predicted_Nf_strength.append(predicted_Nf.cpu().tolist())
    predicted_Sf_strength.append(predicted_Sf.cpu().tolist())
    predicted_Beta_anisotropy.append(predicted_Beta.cpu().tolist())
    true_values.append(true.cpu().tolist())

# %%
true_values_rescaled = []
predicted_Cij_rescaled  = []
predicted_Nf_strength_rescaled  = []
predicted_Sf_strength_rescaled  = []
predicted_Beta_anisotropy_rescaled  = []

y_range = dataset.y_range
y_min = dataset.y_min
for i in range(num_samples):
    true_values_rescaled.append(np.array(true_values)[i,:] * dataset.y_range.numpy() + dataset.y_min.numpy())
    predicted_Cij_rescaled.append(np.array(predicted_Cij)[i] * dataset.y_range[3:].numpy() + dataset.y_min[3:].numpy())
    predicted_Nf_strength_rescaled.append(np.array(predicted_Nf_strength)[i] * dataset.y_range[0].numpy() + dataset.y_min[0].numpy())
    predicted_Sf_strength_rescaled.append(np.array(predicted_Sf_strength)[i] * dataset.y_range[1].numpy() + dataset.y_min[1].numpy())
    predicted_Beta_anisotropy_rescaled.append(np.array(predicted_Beta_anisotropy)[i] * dataset.y_range[2].numpy() + dataset.y_min[2].numpy())

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

y_labels = ['$C_{11}$', '$C_{12}$', '$C_{22}$', '$C_{33}$']
arr = predicted_Cij_rescaled

for i in range(4):
    axes[i].scatter(np.array(true_values_rescaled)[:,i+3], np.array(arr)[:,i])
    
    axes[i].set_xlabel("True")
    axes[i].set_ylabel("Predicted")
    x_minmax = [min(np.array(arr)[:,i])*0.98,max((np.array(arr)[:,i])*1.02)]
    y_minmax = [min(np.array(arr)[:,i])*0.98,max((np.array(arr)[:,i])*1.02)]
    axes[i].plot(x_minmax, y_minmax, linestyle='-.', c='k')
    axes[i].fill_between(x_minmax, np.array(y_minmax) - 0.05 * np.array(y_minmax), np.array(y_minmax) + 0.05 * np.array(y_minmax), alpha=0.4, color='gray') 
    axes[i].fill_between(x_minmax, np.array(y_minmax) - 0.01 * np.array(y_minmax), np.array(y_minmax) + 0.01 * np.array(y_minmax), alpha=0.4, color='gray') 
    axes[i].set_title(y_labels[i])
    axes[i].set_yticks(axes[i].get_xticks())
    axes[i].set_xticks(axes[i].get_yticks())
    axes[i].set_xlim(y_minmax[0]*1.0, y_minmax[1]*1.0)
    axes[i].set_ylim(y_minmax[0]*1.0, y_minmax[1]*1.0)
plt.tight_layout()
pltNameWOExt = f"./logs/{log_name}/cij_error"
fig.savefig(pltNameWOExt+ ".png")
pickle.dump(fig, open(pltNameWOExt + ".pkl", "wb"))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

y_labels = ['Nf Strength', 'Sf Strength', 'Beta Anisotropy']
arr = [predicted_Nf_strength_rescaled, predicted_Sf_strength_rescaled, predicted_Beta_anisotropy_rescaled]

for i in range(3):
    axes[i].scatter(np.array(true_values_rescaled)[:,i], arr[i])
    
    axes[i].set_xlabel("True")
    axes[i].set_ylabel("Predicted")
    x_minmax = [min(np.array(arr)[i])*0.98,max((np.array(arr)[i])*1.02)]
    y_minmax = [min(np.array(arr)[i])*0.98,max((np.array(arr)[i])*1.02)]
    axes[i].plot(x_minmax, y_minmax, linestyle='-.', c='k')
    axes[i].fill_between(x_minmax, np.array(y_minmax) - 0.05 * np.array(y_minmax), np.array(y_minmax) + 0.05 * np.array(y_minmax), alpha=0.4, color='gray') 
    axes[i].fill_between(x_minmax, np.array(y_minmax) - 0.01 * np.array(y_minmax), np.array(y_minmax) + 0.01 * np.array(y_minmax), alpha=0.4, color='gray') 
    axes[i].set_title(y_labels[i])
    axes[i].set_yticks(axes[i].get_xticks())
    axes[i].set_xticks(axes[i].get_yticks())
    axes[i].set_xlim(y_minmax[0]*1.0, y_minmax[1]*1.0)
    axes[i].set_ylim(y_minmax[0]*1.0, y_minmax[1]*1.0)
plt.tight_layout()
pltNameWOExt = f"./logs/{log_name}/scalars_error"
fig.savefig(pltNameWOExt+ ".png")
pickle.dump(fig, open(pltNameWOExt + ".pkl", "wb"))