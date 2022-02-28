import os
import torch
import pickle, csv
import matplotlib.pyplot as plt

##################################################################################################################
uncharact_list = "./dataset/uncharacterized.txt"
with open(uncharact_list, "r") as f:
    skip = ["gdb_" + str(int(x.split()[0])) for x in f.read().split("\n")[9:-2]]

var_names = ["HOMO (eV)", "LUMO (eV)", "GAP (eV)"]
HAR2EV = 27.211386246
datafile = "./dataset/gdb9_gap.csv"
datafile_cut = "./dataset/gdb9_gap_cut.csv"

fileid_list = []
values_list = []
with open(datafile, "r") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    with open(datafile_cut, "w") as file_new:
        csvnewwriter = csv.writer(file_new)
        csvnewwriter.writerow(header)
        for row in csvreader:
            if row[0] in skip:
                continue
            csvnewwriter.writerow(row)
            fileid_list.append(row[0])
            values_list.append([float(x) * HAR2EV for x in row[2:]])
values_tensor = torch.tensor(values_list)
##################################################################################################################
perc_train = 0.9
perc_val = (1 - perc_train) / 2
ntotal = len(fileid_list)
ntrain = int(ntotal * perc_train)
nval = int(ntotal * perc_val)
ntest = ntotal - ntrain - nval
print(ntotal, ntrain, nval, ntest)
randomlist = torch.randperm(ntotal)

idx_train_list = [fileid_list[ifile] for ifile in randomlist[:ntrain]]
idx_val_list = [fileid_list[ifile] for ifile in randomlist[ntrain : ntrain + nval]]
idx_test_list = [fileid_list[ifile] for ifile in randomlist[ntrain + nval :]]

print(idx_train_list)
print(idx_val_list)
print(idx_test_list)

train_tensor = values_tensor[randomlist[:ntrain], :]
val_tensor = values_tensor[randomlist[ntrain : ntrain + nval], :]
test_tensor = values_tensor[randomlist[ntrain + nval :], :]

############################################################

print(torch.max(train_tensor, 0)[0])
print(torch.min(train_tensor, 0)[0])
print(torch.mean(train_tensor, 0))
print(torch.std(train_tensor, 0))

outputfile_name = "qm9_statistics.pkl"
with open(os.path.join("./dataset/", outputfile_name), "wb") as f:
    pickle.dump(torch.max(train_tensor, 0)[0].numpy(), f)
    pickle.dump(torch.min(train_tensor, 0)[0].numpy(), f)
    pickle.dump(torch.mean(train_tensor, 0).numpy(), f)
    pickle.dump(torch.std(train_tensor, 0).numpy(), f)

outputfile_name = "qm9_train_test_val_idx_lists.pkl"
with open(os.path.join("./dataset/", outputfile_name), "wb") as f:
    pickle.dump(idx_train_list, f)
    pickle.dump(idx_val_list, f)
    pickle.dump(idx_test_list, f)
##################################################################################################################
fig, axs = plt.subplots(3, 4, figsize=(15, 12))
plt.subplots_adjust(
    left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.35
)
for idata, dataset, datasetname in zip(
    range(4),
    [values_tensor, train_tensor, val_tensor, test_tensor],
    ["total", "train", "val", "test"],
):
    for ifeat in range(values_tensor.shape[1]):
        ax = axs[ifeat, idata]
        xfeat = dataset[:, ifeat].squeeze().detach().numpy()
        hist1d, bine, _ = ax.hist(xfeat, bins=50)

        if ifeat == 0:
            ax.set_title(datasetname + ", " + str(len(xfeat)))
        ax.set_xlabel(var_names[ifeat])

fig.savefig("./dataset/qm9_train_val_test_hist.png")
plt.close()
