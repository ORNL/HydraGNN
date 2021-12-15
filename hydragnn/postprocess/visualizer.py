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

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from itertools import chain
import time, pickle
import numpy as np
from math import sqrt, floor, ceil

plt.rcParams.update({"font.size": 18})


class Visualizer:
    """A class used for visualizing results of GCNN outputs.

    Methods
    -------
    __init__(self, model_with_config_name: str, node_feature=[], num_heads=1, head_dims=[1])
        Create a Visualizer object with model name (also the location to save the plots), node feature list,
        number of heads, and head dimensions list.
    __hist2d_contour(self, data1, data2)
        Calculate joint histogram of data1 and data2 values.
    __err_condmean(self, data1, data2, weight=1.0)
        Calculate conditional mean values of the weighted difference between data1 and data2 on data1, i.e., <weight*|data1-data2|>_data1.
    __scatter_impl(self,ax,x,y,s=None,c=None,marker=None,title=None,x_label=None,y_label=None,xylim_equal=False,)
        Create scatter plot of x and y values.

    ##create plots for a single variable with name varname
    create_plot_global_analysis(self, varname, true_values, predicted_values, save_plot=True)
        Creates scatter/conditonal mean/error pdf plots from the true and predicted values of variable varname.
        For node level output, statistics across all nodes, l2 length and sum, are also considered for analysis.
        (file: varname +"_scatter_condm_err.png")
    create_scatter_plot_variable(self, varname, true_values, predicted_values, iepoch=None, save_plot=True)
        Creates scatter plots for true_values and predicted values of variable varname at iepoch.
        (file: varname + ".png")
    create_error_histogram_plot_nodes(self, varname, true_values, predicted_values, iepoch=None, save_plot=True)
        Creates error histogram plot for the true and predicted values of variable varname at iepoch.[node level]
        (file: varname+"_error_hist1d_"+ str(iepoch).zfill(4)+ ".png" or varname+"_error_hist1d.png" )
    create_scatter_plot_nodes_vec(self, varname, true_values, predicted_values, iepoch=None, save_plot=True):
        Creates scatter plots for true and predicted values of vector variable varname.
        (file: varname+"_"+ str(iepoch).zfill(4) + ".png" or varname+".png" )

    #create plots for all heads
    plot_history(self,total_loss_train,total_loss_val,total_loss_test,task_loss_train_sum,task_loss_val_sum, task_loss_test_sum,task_loss_train,task_loss_val,task_loss_test,task_weights,task_names,):
        Save history of losses to file and plot.
    create_scatter_plots(self, true_values, predicted_values, output_names=None, iepoch=None)
        Creates scatter plots for all head predictions. One plot for each head
    create_plot_global(self, true_values, predicted_values, output_names=None)
        Creates global analysis for all head predictons, e.g., scatter/condmean/error pdf plot. One plot for each head
    """

    def __init__(
        self,
        model_with_config_name: str,
        node_feature: [],
        num_nodes_list: [],
        num_heads=1,
        head_dims=[1],
    ):
        self.true_values = []
        self.predicted_values = []
        self.model_with_config_name = model_with_config_name
        self.node_feature = node_feature
        self.num_nodes = len(node_feature[0])
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.num_nodes_list = num_nodes_list

    def __hist2d_contour(self, data1, data2):
        hist2d_pasr, xedge_pasr, yedge_pasr = np.histogram2d(
            np.hstack(data1), np.hstack(data2), bins=50
        )
        xcen_pasr = 0.5 * (xedge_pasr[0:-1] + xedge_pasr[1:])
        ycen_pasr = 0.5 * (yedge_pasr[0:-1] + yedge_pasr[1:])
        BCTY_pasr, BCTX_pasr = np.meshgrid(ycen_pasr, xcen_pasr)
        hist2d_pasr = hist2d_pasr / np.amax(hist2d_pasr)
        return BCTX_pasr, BCTY_pasr, hist2d_pasr

    def __err_condmean(self, data1, data2, weight=1.0):
        errabs = np.abs(np.hstack(data1) - np.hstack(data2)) * weight
        hist2d_pasr, xedge_pasr, yedge_pasr = np.histogram2d(
            np.hstack(data1), errabs, bins=50
        )
        xcen_pasr = 0.5 * (xedge_pasr[0:-1] + xedge_pasr[1:])
        ycen_pasr = 0.5 * (yedge_pasr[0:-1] + yedge_pasr[1:])
        hist2d_pasr = hist2d_pasr / np.amax(hist2d_pasr)
        mean1d_cond = np.dot(hist2d_pasr, ycen_pasr) / (
            np.sum(hist2d_pasr, axis=1) + 1e-12
        )
        return xcen_pasr, mean1d_cond

    def __scatter_impl(
        self,
        ax,
        x,
        y,
        s=None,
        c=None,
        marker=None,
        title=None,
        x_label=None,
        y_label=None,
        xylim_equal=False,
    ):
        ax.scatter(x, y, s=s, edgecolor="b", marker=marker, facecolor="none")

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if xylim_equal:
            ax.set_aspect("equal")
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

        self.add_identity(ax, color="r", ls="--")

    def create_plot_global_analysis(
        self, varname, true_values, predicted_values, save_plot=True
    ):
        """Creates scatter/condmean/error pdf plots for the true and predicted values of variable varname."""
        nshape = np.asarray(predicted_values).shape
        if nshape[1] == 1:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
            plt.subplots_adjust(
                left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
            )
            ax = axs[0]
            self.__scatter_impl(
                ax,
                true_values,
                predicted_values,
                title="Scalar output",
                x_label="True",
                y_label="Predicted",
                xylim_equal=True,
            )

            ax = axs[1]
            xtrue, error = self.__err_condmean(
                true_values, predicted_values, weight=1.0
            )
            ax.plot(xtrue, error, "ro")
            ax.set_title("Conditional mean abs. error")
            ax.set_xlabel("True")
            ax.set_ylabel("abs. error")

            ax = axs[2]
            hist1d, bin_edges = np.histogram(
                np.array(predicted_values) - np.array(true_values),
                bins=40,
                density=True,
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            ax.set_title("Scalar output: error PDF")
            ax.set_xlabel("Error")
            ax.set_ylabel("PDF")
        else:
            fig, axs = plt.subplots(3, 3, figsize=(18, 16))
            vlen_true = []
            vlen_pred = []
            vsum_true = []
            vsum_pred = []
            for isamp in range(nshape[0]):
                vlen_true.append(
                    sqrt(sum([comp ** 2 for comp in true_values[isamp][:]]))
                )
                vlen_pred.append(
                    sqrt(sum([comp ** 2 for comp in predicted_values[isamp][:]]))
                )
                vsum_true.append(sum(true_values[isamp][:]))
                vsum_pred.append(sum(predicted_values[isamp][:]))

            ax = axs[0, 0]
            self.__scatter_impl(
                ax,
                vlen_true,
                vlen_pred,
                title="Vector output: length",
                x_label="True",
                y_label="Predicted",
                xylim_equal=True,
            )

            ax = axs[1, 0]
            xtrue, error = self.__err_condmean(
                vlen_true, vlen_pred, weight=1.0 / sqrt(nshape[1])
            )
            ax.plot(xtrue, error, "ro")
            ax.set_ylabel("Conditional mean abs error")
            ax.set_xlabel("True")

            ax = axs[2, 0]
            hist1d, bin_edges = np.histogram(
                np.array(vlen_pred) - np.array(vlen_true), bins=40, density=True
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            ax.set_ylabel("Error PDF")
            ax.set_xlabel("Error")

            ax = axs[0, 1]
            self.__scatter_impl(
                ax,
                vsum_true,
                vsum_pred,
                title="Vector output: sum",
                x_label="True",
                y_label="Predicted",
                xylim_equal=True,
            )

            ax = axs[1, 1]
            xtrue, error = self.__err_condmean(
                vsum_true, vsum_pred, weight=1.0 / nshape[1]
            )
            ax.plot(xtrue, error, "ro")

            ax = axs[2, 1]
            hist1d, bin_edges = np.histogram(
                np.array(vsum_pred) - np.array(vsum_true), bins=40, density=True
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")

            truecomp = []
            predcomp = []
            for icomp in range(nshape[1]):
                truecomp.append(true_values[:][icomp])
                predcomp.append(predicted_values[:][icomp])

            ax = axs[0, 2]
            self.__scatter_impl(
                ax,
                truecomp,
                predcomp,
                title="Vector output: components",
                x_label="True",
                y_label="Predicted",
                xylim_equal=True,
            )

            ax = axs[1, 2]
            xtrue, error = self.__err_condmean(truecomp, predcomp, weight=1.0)
            ax.plot(xtrue, error, "ro")

            ax = axs[2, 2]
            hist1d, bin_edges = np.histogram(
                np.array(predcomp) - np.array(truecomp), bins=40, density=True
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.95, wspace=0.2, hspace=0.25
            )

        if save_plot:
            fig.savefig(
                f"./logs/{self.model_with_config_name}/"
                + varname
                + "_scatter_condm_err.png"
            )
            plt.close()
        else:
            plt.show()

    def create_scatter_plot_variable(
        self, varname, true_values, predicted_values, iepoch=None, save_plot=True
    ):
        """Creates scatter plots for true_values and predicted values of variable varname at iepoch."""

        nshape = np.asarray(predicted_values).shape
        if nshape[1] == 1:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            ax = axs[0]
            self.__scatter_impl(
                ax,
                true_values,
                predicted_values,
                title=varname,
                x_label="True",
                y_label="Predicted",
                xylim_equal=True,
            )

            ax = axs[1]
            hist1d, bin_edges = np.histogram(
                np.array(predicted_values) - np.array(true_values),
                bins=40,
                density=True,
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            ax.set_title(varname + ": error PDF")

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.25
            )
        else:
            nrow = floor(sqrt((nshape[1] + 2)))
            ncol = ceil((nshape[1] + 2) / nrow)
            fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
            axs = axs.flatten()
            for inode in range(nshape[1]):
                xfeature = []
                truecomp = []
                predcomp = []
                for isamp in range(nshape[0]):
                    xfeature.append(self.node_feature[isamp][inode])
                    truecomp.append(true_values[isamp][inode])
                    predcomp.append(predicted_values[isamp][inode])
                ax = axs[inode]
                self.__scatter_impl(
                    ax,
                    truecomp,
                    predcomp,
                    s=6,
                    c=xfeature,
                    title="node:" + str(inode),
                    xylim_equal=True,
                )

            ax = axs[nshape[1]]  # summation over all the nodes/nodes
            xfeature = []
            truecomp = []
            predcomp = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(self.node_feature[isamp][:]))
                truecomp.append(sum(true_values[isamp][:]))
                predcomp.append(sum(predicted_values[isamp][:]))
            self.__scatter_impl(
                ax, truecomp, predcomp, s=40, c=xfeature, title="SUM", xylim_equal=True
            )
            # summation over all the samples for each node
            ax = axs[nshape[1] + 1]
            xfeature = []
            truecomp = []
            predcomp = []
            for inode in range(nshape[1]):
                xfeature.append(sum(self.node_feature[:][inode]))
                truecomp.append(sum(true_values[:][inode]))
                predcomp.append(sum(predicted_values[:][inode]))
            self.__scatter_impl(
                ax,
                truecomp,
                predcomp,
                s=40,
                c=xfeature,
                title="SMP_Mean4sites:0-" + str(nshape[1]),
                xylim_equal=True,
            )

            for iext in range(nshape[1] + 2, axs.size):
                axs[iext].axis("off")
            plt.subplots_adjust(
                left=0.05, bottom=0.05, right=0.98, top=0.95, wspace=0.1, hspace=0.25
            )
        if save_plot:
            if iepoch:
                fig.savefig(
                    f"./logs/{self.model_with_config_name}/"
                    + varname
                    + "_"
                    + str(iepoch).zfill(4)
                    + ".png"
                )
            else:
                fig.savefig(f"./logs/{self.model_with_config_name}/" + varname + ".png")
            plt.close()
        return

    def create_error_histogram_plot_nodes(
        self, varname, true_values, predicted_values, iepoch=None, save_plot=True
    ):
        """Creates error histogram plot for true and predicted values of variable varname at iepoch.[node level]"""

        nshape = np.asarray(predicted_values).shape
        if nshape[1] == 1:
            return
        else:
            nrow = floor(sqrt((nshape[1] + 2)))
            ncol = ceil((nshape[1] + 2) / nrow)
            # error plots
            fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 3.2))
            axs = axs.flatten()
            for inode in range(nshape[1]):
                xfeature = []
                truecomp = []
                predcomp = []
                for isamp in range(nshape[0]):
                    xfeature.append(self.node_feature[isamp][inode])
                    truecomp.append(true_values[isamp][inode])
                    predcomp.append(predicted_values[isamp][inode])
                hist1d, bin_edges = np.histogram(
                    np.array(predcomp) - np.array(truecomp), bins=40, density=True
                )
                ax = axs[inode]
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
                ax.set_title("node:" + str(inode))

            ax = axs[nshape[1]]
            xfeature = []
            truecomp = []
            predcomp = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(self.node_feature[isamp][:]))
                truecomp.append(sum(true_values[isamp][:]))
                predcomp.append(sum(predicted_values[isamp][:]))
            hist1d, bin_edges = np.histogram(
                np.array(predcomp) - np.array(truecomp), bins=40, density=True
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            ax.set_title("SUM")

            # summation over all the samples for each node
            ax = axs[nshape[1] + 1]
            xfeature = []
            truecomp = []
            predcomp = []
            for inode in range(nshape[1]):
                xfeature.append(sum(self.node_feature[:][inode]))
                truecomp.append(sum(true_values[:][inode]))
                predcomp.append(sum(predicted_values[:][inode]))
            hist1d, bin_edges = np.histogram(
                np.array(predcomp) - np.array(truecomp), bins=40, density=True
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            ax.set_title("SMP_Mean4sites:0-" + str(nshape[1]))

            for iext in range(nshape[1] + 2, axs.size):
                axs[iext].axis("off")
            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.35
            )
            if save_plot:
                if iepoch:
                    fig.savefig(
                        f"./logs/{self.model_with_config_name}/"
                        + varname
                        + "_error_hist1d_"
                        + str(iepoch).zfill(4)
                        + ".png"
                    )
                else:
                    fig.savefig(
                        f"./logs/{self.model_with_config_name}/"
                        + varname
                        + "_error_hist1d.png"
                    )
                plt.close()

    def create_scatter_plot_nodes_vec(
        self, varname, true_values, predicted_values, iepoch=None, save_plot=True
    ):
        """Creates scatter plots for true and predicted values of vector variable varname[nodel level]."""

        nshape = np.asarray(predicted_values).shape
        predicted_vec = np.reshape(np.asarray(predicted_values), (nshape[0], -1, 3))
        true_vec = np.reshape(np.asarray(true_values), (nshape[0], -1, 3))
        num_nodes = true_vec.shape[1]

        markers_vec = ["o", "s", "d"]  # different markers for three vector components
        nrow = floor(sqrt((num_nodes + 2)))
        ncol = ceil((num_nodes + 2) / nrow)
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
        axs = axs.flatten()
        for inode in range(num_nodes):
            for icomp in range(3):
                xfeature = []
                truecomp = []
                predcomp = []
                for isamp in range(nshape[0]):
                    xfeature.append(self.node_feature[isamp][inode])
                    truecomp.append(true_vec[isamp, inode, icomp])
                    predcomp.append(predicted_vec[isamp, inode, icomp])
                ax = axs[inode]
                self.__scatter_impl(
                    ax,
                    truecomp,
                    predcomp,
                    s=6,
                    c=xfeature,
                    marker=markers_vec[icomp],
                    title="node:" + str(inode),
                    xylim_equal=True,
                )

        ax = axs[num_nodes]  # summation over all the nodes/nodes
        for icomp in range(3):
            xfeature = []
            truecomp = []
            predcomp = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(self.node_feature[isamp][:]))
                truecomp.append(sum(true_vec[isamp, :, icomp]))
                predcomp.append(sum(predicted_vec[isamp, :, icomp]))
            self.__scatter_impl(
                ax,
                truecomp,
                predcomp,
                s=40,
                c=xfeature,
                marker=markers_vec[icomp],
                title="SUM",
                xylim_equal=True,
            )

        ax = axs[num_nodes + 1]  # summation over all the samples for each node
        for icomp in range(3):
            xfeature = []
            truecomp = []
            predcomp = []
            for inode in range(num_nodes):
                xfeature.append(sum(self.node_feature[:][inode]))
                truecomp.append(sum(true_vec[:, inode, icomp]))
                predcomp.append(sum(predicted_vec[:, inode, icomp]))
            self.__scatter_impl(
                ax,
                truecomp,
                predcomp,
                s=40,
                c=xfeature,
                marker=markers_vec[icomp],
                title="SMP_Mean4sites:0-" + str(num_nodes),
                xylim_equal=True,
            )

        for iext in range(num_nodes + 2, axs.size):
            axs[iext].axis("off")
        plt.subplots_adjust(
            left=0.05, bottom=0.05, right=0.98, top=0.95, wspace=0.1, hspace=0.25
        )

        if save_plot:
            if iepoch:
                fig.savefig(
                    f"./logs/{self.model_with_config_name}/"
                    + varname
                    + "_"
                    + str(iepoch).zfill(4)
                    + ".png"
                )
            else:
                fig.savefig(f"./logs/{self.model_with_config_name}/" + varname + ".png")
            plt.close()

    def add_identity(self, axes, *line_args, **line_kwargs):
        (identity,) = axes.plot([], [], *line_args, **line_kwargs)

        def callback(axes):
            low_x, high_x = axes.get_xlim()
            low_y, high_y = axes.get_ylim()
            low = max(low_x, low_y)
            high = min(high_x, high_y)
            identity.set_data([low, high], [low, high])

        callback(axes)
        axes.callbacks.connect("xlim_changed", callback)
        axes.callbacks.connect("ylim_changed", callback)
        return axes

    def plot_history(
        self,
        total_loss_train,
        total_loss_val,
        total_loss_test,
        task_loss_train,
        task_loss_val,
        task_loss_test,
        task_weights,
        task_names,
    ):
        nrow = 1
        fhist = open(f"./logs/{self.model_with_config_name}/history_loss.pckl", "wb")
        pickle.dump(
            [
                total_loss_train,
                total_loss_val,
                total_loss_test,
                task_loss_train,
                task_loss_val,
                task_loss_test,
                task_weights,
                task_names,
            ],
            fhist,
        )
        fhist.close()
        num_tasks = len(task_loss_train[0])
        if num_tasks > 0:
            task_loss_train = np.array(task_loss_train)
            task_loss_val = np.array(task_loss_val)
            task_loss_test = np.array(task_loss_test)
            nrow = 2
        fig, axs = plt.subplots(nrow, num_tasks, figsize=(16, 6 * nrow))
        axs = axs.flatten()
        ax = axs[0]
        ax.plot(total_loss_train, "-", label="train")
        ax.plot(total_loss_val, ":", label="validation")
        ax.plot(total_loss_test, "--", label="test")
        ax.set_title("total loss")
        ax.set_xlabel("Epochs")
        ax.set_yscale("log")
        ax.legend()
        for iext in range(1, num_tasks):
            axs[iext].axis("off")
        for ivar in range(task_loss_train.shape[1]):
            ax = axs[num_tasks + ivar]
            ax.plot(task_loss_train[:, ivar], label="train")
            ax.plot(task_loss_val[:, ivar], label="validation")
            ax.plot(task_loss_test[:, ivar], "--", label="test")
            ax.set_title(task_names[ivar] + ", {:.4f}".format(task_weights[ivar]))
            ax.set_xlabel("Epochs")
            ax.set_yscale("log")
            if ivar == 0:
                ax.legend()
        for iext in range(num_tasks + task_loss_train.shape[1], axs.size):
            axs[iext].axis("off")
        plt.subplots_adjust(
            left=0.1, bottom=0.08, right=0.98, top=0.9, wspace=0.25, hspace=0.3
        )
        fig.savefig(f"./logs/{self.model_with_config_name}/history_loss.png")
        plt.close()

    def create_scatter_plots(
        self, true_values, predicted_values, output_names=None, iepoch=None
    ):
        """Creates scatter plots for all head predictions."""
        for ihead in range(self.num_heads):
            if self.head_dims[ihead] == 3:
                # vector output
                self.create_scatter_plot_nodes_vec(
                    output_names[ihead],
                    true_values[ihead],
                    predicted_values[ihead],
                    iepoch,
                )
            else:
                self.create_scatter_plot_variable(
                    output_names[ihead],
                    true_values[ihead],
                    predicted_values[ihead],
                    iepoch,
                )
                self.create_error_histogram_plot_nodes(
                    output_names[ihead],
                    true_values[ihead],
                    predicted_values[ihead],
                    iepoch,
                )

    def create_plot_global(self, true_values, predicted_values, output_names=None):
        """Creates global analysis for all head predictons, e.g., scatter/condmean/error pdf plot."""
        for ihead in range(self.num_heads):
            self.create_plot_global_analysis(
                output_names[ihead],
                true_values[ihead],
                predicted_values[ihead],
                save_plot=True,
            )

    def num_nodes_plot(
        self,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.hist(self.num_nodes_list)
        ax.set_title("Histogram of graph size in test set")
        ax.set_xlabel("number of nodes")
        fig.savefig(f"./logs/{self.model_with_config_name}/num_nodes.png")
        plt.close()
