import os, json
import logging
import torch
import numpy as np
import hydragnn
from hydragnn.utils.model import print_model
from hydragnn.utils.print_utils import log, iterate_tqdm
from hydragnn.train import get_head_indices, reduce_values_ranks, gather_tensor_ranks


class model_ensemble(torch.nn.Module):
    def __init__(self, model_dir_list, modelname=None, dir_extra="", verbosity=1):
        super(model_ensemble, self).__init__()
        self.model_dir_list = model_dir_list 
        self.model_ens = torch.nn.ModuleList()
        for imodel, modeldir in enumerate(self.model_dir_list):
            input_filename = os.path.join(modeldir, "config.json")
            with open(input_filename, "r") as f:
                config = json.load(f)
            model = hydragnn.models.create_model_config(
                config=config["NeuralNetwork"],
                verbosity=verbosity,
            )
            model = hydragnn.utils.get_distributed_model(model, verbosity)
            # Print details of neural network architecture
            print("Loading model %d, %s"%(imodel, modeldir))
            print_model(model)
            if modelname is None:
                hydragnn.utils.load_existing_model(model, os.path.basename(os.path.normpath(modeldir)), path=modeldir+"/../"+dir_extra)
            else:
                hydragnn.utils.load_existing_model(model, modelname, path=modeldir+dir_extra)
            self.model_ens.append(model)
        self.num_heads = self.model_ens[0].module.num_heads
        self.loss = self.model_ens[0].module.loss
        self.model_size = len(self.model_dir_list)
    def forward(self, x, meanstd=False):
        y_ens=[]
        for model in self.model_ens:
            y_ens.append(model(x))
        if meanstd:
            head_pred_mean = []
            head_pred_std = []
            for ihead in range(self.num_heads):
                head_pred = []
                for imodel in range(self.model_size):
                    head_pred.append(y_ens[imodel][ihead])
                head_pred_ens = torch.stack(head_pred, dim=0).squeeze()
                head_pred_mean.append(head_pred_ens.mean(axis=0))
                head_pred_std.append(head_pred_ens.std(axis=0))
            return head_pred_mean, head_pred_std
        return y_ens

    def __len__(self):
        return self.model_size

    ##################################################################################################################
def debug_nan(x, message=""):
    try:
        if torch.isnan(x).any():
            print(f"NAN detected in prediction, after {message}: ",x)
            return True
    except:
        if torch.isnan(torch.tensor(x)).any():
            print(f"NAN detected in prediction, after {message}: ",x)
            return True

    return False

def test_ens_GFM(model_ens, loader, verbosity, num_samples=None, saveresultsto=None):
    n_ens=len(model_ens.module)
    num_heads=model_ens.module.num_heads

    num_samples_total = 0
    device = next(model_ens.parameters()).device

    total_error = torch.zeros(n_ens, device=device)
    tasks_error = torch.zeros(n_ens, num_heads, device=device)

    true_values = [[] for _ in range(num_heads)]
    predicted_values = [[[] for _ in range(num_heads)] for _ in range(n_ens)]
   
    for data in iterate_tqdm(loader, verbosity):
        data=data.to(device)
        head_index = get_head_indices(model_ens.module.model_ens[0], data)
        ###########################
        pred_ens = model_ens(data)
        ###########################
        ytrue = data.y
        for ihead in range(num_heads):
            head_val = ytrue[head_index[ihead]]
            true_values[ihead].extend(head_val)
            _ = debug_nan(true_values[ihead], message="true_values %d"%ihead)
        ###########################
        for imodel, pred in enumerate(pred_ens):
            error, tasks_rmse = model_ens.module.loss(pred, data.y, head_index)
            total_error[imodel] += error.item() * data.num_graphs
            if imodel==0:
                num_samples_total += data.num_graphs
            for itask in range(len(tasks_rmse)):
                tasks_error[imodel, itask] += tasks_rmse[itask].item() * data.num_graphs
            for ihead in range(num_heads):
                head_pre = pred[ihead].reshape(-1, 1)
                pred_shape = head_pre.shape
                predicted_values[imodel][ihead].extend(head_pre.tolist())
        if num_samples is not None and num_samples_total > num_samples//hydragnn.utils.get_comm_size_and_rank()[0]-1:
            break
    
    total_error = reduce_values_ranks(total_error)
    tasks_error = reduce_values_ranks(tasks_error)
    num_samples_total = reduce_values_ranks(torch.tensor(num_samples_total)).item()

    predicted_mean= [[] for _ in range(num_heads)]
    predicted_std= [[] for _ in range(num_heads)]
    for ihead in range(num_heads):
        head_pred = []
        print("For head %d"%ihead)
        for imodel in range(len(model_ens.module)):
            head_all = torch.tensor(predicted_values[imodel][ihead])
            head_all = gather_tensor_ranks(head_all)
            print("imodel %d"%imodel, head_all.size())
            if debug_nan(head_all, message="pred from model %d"%imodel):
                print("Warning: NAN detected in model %d; prediction skipped"%imodel)
                continue
            head_pred.append(head_all)
        true_values[ihead] = torch.cat(true_values[ihead], dim=0)
        head_pred_ens = torch.stack(head_pred, dim=0).squeeze()
        head_pred_mean = head_pred_ens.mean(axis=0)
        head_pred_std = head_pred_ens.std(axis=0)
        true_values[ihead] = gather_tensor_ranks(true_values[ihead])
        predicted_mean[ihead] = head_pred_mean 
        predicted_std[ihead] = head_pred_std
        print(head_pred_ens.size(), true_values[ihead].size())
        if saveresultsto is not None and hydragnn.utils.get_comm_size_and_rank()[1]==0:
            m = {'true': true_values[ihead], 'pred_ens': head_pred_ens}
            torch.save(m, saveresultsto +"head%d_%s.db"%(ihead, str(hydragnn.utils.get_comm_size_and_rank()[1])))

    return (
        [tot_err.item() / num_samples_total for tot_err  in total_error],
        [task_err / num_samples_total for task_err in tasks_error],
        true_values,
        predicted_mean,
        predicted_std
    )
