##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import json, os
import sys
import logging
import pickle
from tqdm import tqdm
from mpi4py import MPI
import argparse

import torch
import numpy as np

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print_utils import print_distributed
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.config_utils import (
    update_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

from scipy.interpolate import griddata

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from scipy.interpolate import BSpline, make_interp_spline
import adios2 as ad2

import matplotlib.pyplot as plt
from ensemble_utils import model_ensemble, test_ens_GFM, debug_nan
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.print_utils import log
from hydragnn.utils import nsplit
from mpl_toolkits.axes_grid1 import make_axes_locatable




plt.rcParams.update({"font.size": 20})


def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    ##################################################################################################################
    parser = argparse.ArgumentParser()
    print("gfm starting")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--models_dir_folder", help="folder of trained models", type=str, default=None)
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--log", help="log name", default="GFM_EnsembleInference")
    parser.add_argument("--dataname", help="name of datasets folder", type=str, default="GFM_dataset")
    parser.add_argument("--multi_model_list", help="multidataset list", default="OC2020")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--multi", help="Multi dataset",action="store_const",dest="format",const="multi")
    args = parser.parse_args()
    args.parameters = vars(args)
    assert torch.cuda.is_available()
    ##################################################################################################################
    modeldirlists = args.models_dir_folder.split(",")
    assert len(modeldirlists)==1 or len(modeldirlists)==2
    if len(modeldirlists)==1:
        modeldirlist = [os.path.join(args.models_dir_folder, name) for name in os.listdir(args.models_dir_folder) if os.path.isdir(os.path.join(args.models_dir_folder, name))]
    else:
        modeldirlist = []
        for models_dir_folder in modeldirlists:
            modeldirlist.extend([os.path.join(models_dir_folder, name) for name in os.listdir(models_dir_folder) if os.path.isdir(os.path.join(models_dir_folder, name))])

    var_config = None
    for modeldir in modeldirlist:
        input_filename = os.path.join(modeldir, "config.json")
        with open(input_filename, "r") as f:
            config = json.load(f)
        if var_config is not None:
            assert var_config==config["NeuralNetwork"]["Variables_of_interest"], "Inconsistent variable config in %s"%input_filename
        else:
            var_config = config["NeuralNetwork"]["Variables_of_interest"]
    verbosity=config["Verbosity"]["level"]
    log_name = "GFM_EnsembleInference" if args.log is None else args.log
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    print("ddstore, ddstore_width", args.ddstore, args.ddstore_width)
    ##################################################################################################################
    comm = MPI.COMM_WORLD
    timer = Timer("load_data")
    timer.start()
    if args.format == "multi":
        ## Reading multiple datasets, which requires the following arguments:
        ## --multi_model_list: the list datasets/model names
        modellist = args.multi_model_list.split(",")
        print("datasets list: ", modellist)
        if rank == 0:
            ndata_list = list()
            pna_deg_list = list()
            for model in modellist:
                fname = os.path.join(os.path.dirname(__file__), "./dataset/%s/%s.bp" % (args.dataname, model))
                with ad2.open(fname, "r", MPI.COMM_SELF) as f:
                    f.__next__()
                    ndata = f.read_attribute("trainset/ndata").item()
                    attrs = f.available_attributes()
                    pna_deg = None
                    if "pna_deg" in attrs:
                        pna_deg = f.read_attribute("pna_deg")
                    ndata_list.append(ndata)
                    pna_deg_list.append(pna_deg)
            ndata_list = np.array(ndata_list, dtype=np.float32)
            process_list = np.ceil(ndata_list / sum(ndata_list) * comm_size).astype(
                np.int32
            )
            imax = np.argmax(process_list)
            process_list[imax] = process_list[imax] - (np.sum(process_list) - comm_size)
            process_list = process_list.tolist()

            ## Merge pna_deg using interpolation
            intp_list = list()
            mlen = min([len(pna_deg) for pna_deg in pna_deg_list])
            for pna_deg in pna_deg_list:
                x = np.linspace(0, 1, num=len(pna_deg))
                intp = make_interp_spline(x, pna_deg)
                intp_list.append(intp)

            new_pna_deg_list = list()
            for intp in intp_list:
                x = np.linspace(0, 1, num=mlen)
                y = intp(x)
                new_pna_deg_list.append(y)

            pna_deg = np.zeros_like(new_pna_deg_list[0])
            for new_pna_deg in new_pna_deg_list:
                pna_deg += new_pna_deg
            pna_deg = pna_deg.astype(np.int64).tolist()
        else:
            process_list = None
            pna_deg = None
        process_list = comm.bcast(process_list, root=0)
        pna_deg = comm.bcast(pna_deg, root=0)
        assert comm_size == sum(process_list)

        colorlist = list()
        color = 0
        for n in process_list:
            for _ in range(n):
                colorlist.append(color)
            color += 1
        if rank == 0:
            print("process_list:", process_list)
            print("colorlist:", colorlist)
        mycolor = colorlist[rank]
        mymodel = modellist[mycolor]

        local_comm = comm.Split(mycolor, rank)
        local_comm_rank = local_comm.Get_rank()
        local_comm_size = local_comm.Get_size()

        ## FIXME: Hard-coded for now. Need to find common variable names
        common_variable_names = [
            "x",
            "edge_index",
            "edge_attr",
            "pos",
            "y",
        ]
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s/%s.bp" % (args.dataname, mymodel))
        trainset = AdiosDataset(
            fname,
            "trainset",
            local_comm,
            var_config=var_config,
            keys=common_variable_names,
        )
        valset = AdiosDataset(
            fname,
            "valset",
            local_comm,
            var_config=var_config,
            keys=common_variable_names,
        )
        testset = AdiosDataset(
            fname,
            "testset",
            local_comm,
            var_config=var_config,
            keys=common_variable_names,
        )

        ## Set local set
        for dataset in [trainset, valset, testset]:
            rx = list(nsplit(range(len(dataset)), local_comm_size))[local_comm_rank]
            dataset.setkeys(common_variable_names)
            dataset.setsubset(rx[0], rx[-1] + 1, preload=True)

        if local_comm_rank == 0:
            print(
                rank,
                "color, moddelname, comm size, local size(trainset,valset,testset):",
                mycolor,
                mymodel,
                local_comm_size,
                len(trainset),
                len(valset),
                len(testset),
            )

        assert args.ddstore, "Always use ddstore"
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width, "local": True}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.pna_deg = pna_deg
            valset.pna_deg = pna_deg
            testset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"
    ##################################################################################################################
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    ##################################################################################################################
    model_ens = model_ensemble(modeldirlist)
    model_ens = hydragnn.utils.get_distributed_model(model_ens, verbosity)
    model_ens.eval()
    ##################################################################################################################
    nheads = len(config["NeuralNetwork"]["Variables_of_interest"]["output_names"])
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol, (loader, setname) in enumerate(zip([train_loader, val_loader, test_loader], ["train", "val", "test"])):
        error, rmse_task, true_values, predicted_mean, predicted_uncertainty = test_ens_GFM(model_ens, loader, verbosity, num_samples=1000)
        print_distributed(verbosity,"number of heads %d"%len(true_values))
        print_distributed(verbosity,"number of samples %d"%len(true_values[0]))
        if hydragnn.utils.get_comm_size_and_rank()[1]==0:
            print(setname, "loss=", error, rmse_task)
        assert len(true_values)==len(predicted_mean), "inconsistent number of heads, %d!=%d"%(len(true_values),len(len(predicted_mean)))
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )): 
            _ = debug_nan(true_values[ihead], message="checking on true for %s"%output_name)
            _ = debug_nan(predicted_mean[ihead], message="checking on predicted mean for %s"%output_name)
            head_true = true_values[ihead].cpu().squeeze().numpy() 
            head_pred = predicted_mean[ihead].cpu().squeeze().numpy() 
            head_uncertainty = predicted_uncertainty[ihead].cpu().squeeze().numpy()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]

            try:
                ax = axs[ihead, icol]
            except:
                ax = axs[icol]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                print(setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
            hist2d_norm = getcolordensity(head_true, head_pred)
            #ax.errorbar(head_true, head_pred, yerr=head_uncertainty, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(head_true, head_pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname, fontsize=24)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2e}".format(error_mae),
            )
            if icol==0:
                ax.set_ylabel("Predicted")
            if ihead==1:
                ax.set_xlabel("True")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar=fig.colorbar(sc, cax=cax, orientation='vertical')
            ax.set_aspect('equal', adjustable='box')
            xmin, xmax = ax.get_ylim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(min(xmin, ymin), max(xmax,ymax))
            ax.set_ylim(min(xmin, ymin), max(xmax,ymax))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.4, hspace=0.3)
    fig.savefig("./logs/" + log_name + "/parity_plot_all"+'-'.join(modellist)+".png",dpi=300)
    plt.close()
    
    fig, axs = plt.subplots(nheads, 3, figsize=(18, 6*nheads))
    for icol, (loader, setname) in enumerate(zip([train_loader, val_loader, test_loader], ["train", "val", "test"])):
        error, rmse_task, true_values, predicted_mean, predicted_uncertainty = test_ens_GFM(model_ens, loader, verbosity, num_samples=4096, saveresultsto=f"./logs/{log_name}/{'-'.join(modellist)}_{setname}_")
        print_distributed(verbosity,"number of heads %d"%len(true_values))
        print_distributed(verbosity,"number of samples %d"%len(true_values[0]))
        if hydragnn.utils.get_comm_size_and_rank()[1]==0:
            print(setname, "loss=", error, rmse_task)
        assert len(true_values)==len(predicted_mean), "inconsistent number of heads, %d!=%d"%(len(true_values),len(len(predicted_mean)))
        for  ihead, (output_name, output_type, output_dim) in enumerate(zip(
        config["NeuralNetwork"]["Variables_of_interest"]["output_names"],
        config["NeuralNetwork"]["Variables_of_interest"]["type"],
        config["NeuralNetwork"]["Variables_of_interest"]["output_dim"],
        )):
            _ = debug_nan(true_values[ihead], message="checking on true for %s"%output_name)
            _ = debug_nan(predicted_mean[ihead], message="checking on predicted mean for %s"%output_name)
            head_true = true_values[ihead].cpu().squeeze().numpy() 
            head_pred = predicted_mean[ihead].cpu().squeeze().numpy() 
            head_uncertainty = predicted_uncertainty[ihead].cpu().squeeze().numpy() 
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = var_config["output_names"][ihead]

            try:
                ax = axs[ihead, icol]
            except:
                ax = axs[icol]
            error_mae = np.mean(np.abs(head_pred - head_true))
            error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
            if hydragnn.utils.get_comm_size_and_rank()[1]==0:
                print(setname, varname, ": mae=", error_mae, ", rmse= ", error_rmse)
            hist2d_norm = getcolordensity(head_true, head_pred)
            ax.errorbar(head_true, head_pred, yerr=head_uncertainty, fmt = '', linewidth=0.5, ecolor="b", markerfacecolor="none", ls='none')
            sc=ax.scatter(head_true, head_pred, s=12, c=hist2d_norm, vmin=0, vmax=1)
            minv = np.minimum(np.amin(head_pred), np.amin(head_true))
            maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
            ax.plot([minv, maxv], [minv, maxv], "r--")
            ax.set_title(setname + "; " + varname, fontsize=24)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2e}".format(error_mae),
            )
            if icol==0:
                ax.set_ylabel("Predicted")
            if ihead==1:
                ax.set_xlabel("True")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar=fig.colorbar(sc, cax=cax, orientation='vertical')
            ax.set_aspect('equal', adjustable='box')
            xmin, xmax = ax.get_ylim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(min(xmin, ymin), max(xmax,ymax))
            ax.set_ylim(min(xmin, ymin), max(xmax,ymax))
    fig.savefig("./logs/" + log_name + "/parity_plot_uncertainty_"+'-'.join(modellist)+".png",dpi=300)
    plt.close()
    ##################################################################################################################
