import os
import torch
import pickle, csv
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from ogb_utils import *

# from multiprocessing import Pool
import torch.multiprocessing as mp
import time, math
import copy

##################################################################################################################
def smiles2graphlist(
    smileset, valueset, indexlists, set_flag="train", filedir="dataset/processed"
):
    rank = mp.current_process()._identity[0] - 1
    print("rank=", rank)
    print(len(indexlists))
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    filename = os.path.join(filedir, "train_" + str(rank) + ".pt")
    smileset_ = [smileset[isample] for isample in indexlists]
    valueset_ = [valueset[isample] for isample in indexlists]
    print("Hello from " + str(rank) + "; length=" + str(len(smileset_)))
    dataset_list = []
    for smilestr, ytarget in zip(smileset_, valueset_):
        dataset_list.append(generate_graphdata(smilestr, ytarget))
    print("Done from " + str(rank) + "; length=" + str(len(dataset_list)))
    torch.save(dataset_list, filename)


##################################################################################################################
##################################################################################################################
if __name__ == "__main__":
    var_names = ["GAP"]
    datafile_processed = os.path.join(
        os.path.dirname(__file__), "dataset/pcqm4m_gap.pt"
    )
    datafile = os.path.join(os.path.dirname(__file__), "dataset/pcqm4m_gap.csv")
    smiles_sets, values_sets = datasets_load(datafile)
    ##################################################################################################################
    smileset = smiles_sets[0]
    valueset = values_sets[0]
    print("len(smileset)=", len(smileset))
    start_time = time.time()
    num_proc = max(1, math.ceil(os.cpu_count() * 0.7))
    nsamp = len(smileset)
    ind_lists = [None] * num_proc
    nsamp_perproc = nsamp // num_proc
    istart = 0
    for iproc in range(num_proc):
        print(iproc, len(ind_lists))
        iend = istart + nsamp_perproc if iproc < num_proc - 1 else nsamp
        ind_lists[iproc] = [isample for isample in range(istart, iend)]
        print(iproc, istart, iend)
        istart = iend
    print(len(ind_lists), len(ind_lists[0]))
    p = mp.Pool(num_proc, maxtasksperchild=1)
    p.starmap(
        smiles2graphlist, zip([smileset] * num_proc, [valueset] * num_proc, ind_lists)
    )
    p.close()
    p.join()
