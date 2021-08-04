import matplotlib.pyplot as plt
from itertools import chain
import time, pickle
import numpy as np
from math import sqrt, floor, ceil

plt.rcParams.update({"font.size": 18})


class Visualizer:
    """A class used for visualizing values in a scatter plot. There are two attributes: true_values and predicted_values that we want to see
    in a scatter plot. The ideal case is that the values will be positioned on a thin diagonal of the scatter plot.

    Methods
    -------
    add_test_values(true_values: [], predicted_values: [])
        Add the true and predicted values to the lists.
    create_scatter_plot()
        Create the scatter plot out of true and predicted values.
    """

    def __init__(self, model_with_config_name: str):
        self.true_values = []
        self.predicted_values = []
        self.model_with_config_name = model_with_config_name

    def add_test_values(self, true_values: [], predicted_values: []):
        """Append true and predicted values to existing lists.

        Parameters
        ----------
        true_values: []
            List of true values to append to existing one.
        predicted_values: []
            List of predicted values to append to existing one.
        """
        self.true_values.extend(true_values)
        self.predicted_values.extend(predicted_values)

    def __convert_to_list(self):
        """When called it performs flattening of a list because the values that are stored in true and predicted values lists are in
        the shape: [[1], [2], ...] and in order to visualize them in scatter plot they need to be in the shape: [1, 2, ...].
        """
        if len(self.true_values) * len(self.true_values[0]) != len(
            self.predicted_values
        ) * len(self.predicted_values[0]):
            print("Length of true and predicted values array is not the same!")

        self.true_values = list(chain.from_iterable(self.true_values))
        self.predicted_values = list(chain.from_iterable(self.predicted_values))

    def hist2d_contour(self, data1, data2):
        hist2d_pasr, xedge_pasr, yedge_pasr = np.histogram2d(
            np.hstack(data1), np.hstack(data2), bins=50
        )
        xcen_pasr = 0.5 * (xedge_pasr[0:-1] + xedge_pasr[1:])
        ycen_pasr = 0.5 * (yedge_pasr[0:-1] + yedge_pasr[1:])
        BCTY_pasr, BCTX_pasr = np.meshgrid(ycen_pasr, xcen_pasr)
        hist2d_pasr = hist2d_pasr / np.amax(hist2d_pasr)
        return BCTX_pasr, BCTY_pasr, hist2d_pasr

    def hist1d_err(self, data1, data2, weight=1.0):
        errabs = np.abs(np.hstack(data1) - np.hstack(data2)) * weight
        hist2d_pasr, xedge_pasr, yedge_pasr = np.histogram2d(
            np.hstack(data1), errabs, bins=50
        )
        xcen_pasr = 0.5 * (xedge_pasr[0:-1] + xedge_pasr[1:])
        ycen_pasr = 0.5 * (yedge_pasr[0:-1] + yedge_pasr[1:])
        hist2d_pasr = hist2d_pasr / np.amax(hist2d_pasr)
        mean1d_cond = np.dot(hist2d_pasr, ycen_pasr) / np.sum(hist2d_pasr, axis=1)
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
        ax.scatter(x, y, s, c, marker)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if xylim_equal:
            ax.set_aspect("equal")
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

    def create_plot_global(self, varname, save_plot=True):
        """Creates global scatter/condmean/error pdf plot from stored values in the true and  predicted values lists."""
        nshape = np.asarray(self.predicted_values).shape
        if nshape[1] == 1:
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
            plt.subplots_adjust(
                left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
            )
            ax = axs[0]
            self.__scatter_impl(
                ax,
                self.true_values,
                self.predicted_values,
                title="Scalar output",
                x_label="True",
                y_label="Predicted",
                xylim_equal=True,
            )

            ax = axs[1]
            xtrue, error = self.hist1d_err(
                self.true_values, self.predicted_values, weight=1.0
            )
            ax.plot(xtrue, error, "ro")
            ax.set_title("Conditional mean abs. error")
            ax.set_xlabel("True")
            ax.set_ylabel("abs. error")

            ax = axs[2]
            hist1d, bin_edges = np.histogram(
                np.array(self.predicted_values) - np.array(self.true_values),
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
                    sqrt(sum([comp ** 2 for comp in self.true_values[isamp][:]]))
                )
                vlen_pred.append(
                    sqrt(sum([comp ** 2 for comp in self.predicted_values[isamp][:]]))
                )
                vsum_true.append(sum(self.true_values[isamp][:]))
                vsum_pred.append(sum(self.predicted_values[isamp][:]))

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
            xtrue, error = self.hist1d_err(
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
            xtrue, error = self.hist1d_err(vsum_true, vsum_pred, weight=1.0 / nshape[1])
            ax.plot(xtrue, error, "ro")

            ax = axs[2, 1]
            hist1d, bin_edges = np.histogram(
                np.array(vsum_pred) - np.array(vsum_true), bins=40, density=True
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")

            truecomp = []
            predcomp = []
            for icomp in range(nshape[1]):
                truecomp.append(self.true_values[:][icomp])
                predcomp.append(self.predicted_values[:][icomp])

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
            xtrue, error = self.hist1d_err(truecomp, predcomp, weight=1.0)
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

    def create_scatter_plot_atoms(
        self, varname, x_atomfeature, iepoch=None, save_plot=True
    ):
        """Creates scatter plots for scalar output and vector outputs."""

        nshape = np.asarray(self.predicted_values).shape
        if nshape[1] == 1:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            ax = axs[0]
            self.__scatter_impl(
                ax,
                self.true_values,
                self.predicted_values,
                title=varname,
                x_label="True",
                y_label="Predicted",
                xylim_equal=True,
            )

            ax = axs[1]
            hist1d, bin_edges = np.histogram(
                np.array(self.predicted_values) - np.array(self.true_values),
                bins=40,
                density=True,
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            ax.set_title(varname + ": error PDF")

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.25
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
                    fig.savefig(
                        f"./logs/{self.model_with_config_name}/" + varname + ".png"
                    )
                plt.close()
            return
        else:
            nrow = floor(sqrt((nshape[1] + 1)))
            ncol = ceil((nshape[1] + 1) / nrow)
            fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
            axs = axs.flatten()
            for iatom in range(nshape[1]):
                xfeature = []
                truecomp = []
                predcomp = []
                for isamp in range(nshape[0]):
                    xfeature.append(x_atomfeature[isamp][iatom])
                    truecomp.append(self.true_values[isamp][iatom])
                    predcomp.append(self.predicted_values[isamp][iatom])
                ax = axs[iatom]
                self.__scatter_impl(
                    ax,
                    truecomp,
                    predcomp,
                    s=6,
                    c=xfeature,
                    title="atom:" + str(iatom),
                    xylim_equal=True,
                )

            ax = axs[nshape[1]]  # summation over all the atoms/nodes
            xfeature = []
            truecomp = []
            predcomp = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(x_atomfeature[isamp][:]))
                truecomp.append(sum(self.true_values[isamp][:]))
                predcomp.append(sum(self.predicted_values[isamp][:]))
            self.__scatter_impl(
                ax, truecomp, predcomp, s=40, c=xfeature, title="SUM", xylim_equal=True
            )

            ax = axs[nshape[1] + 1]  # summation over all the samples for each atom/node
            xfeature = []
            truecomp = []
            predcomp = []
            for iatom in range(nshape[1]):
                xfeature.append(sum(x_atomfeature[:][iatom]))
                truecomp.append(sum(self.true_values[:][iatom]))
                predcomp.append(sum(self.predicted_values[:][iatom]))
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
                    fig.savefig(
                        f"./logs/{self.model_with_config_name}/" + varname + ".png"
                    )
                plt.close()

    def create_error_histogram_plot_atoms(
        self, varname, x_atomfeature, iepoch=None, save_plot=True
    ):
        """Creates error histogram plot for vector outputs."""

        nshape = np.asarray(self.predicted_values).shape
        if nshape[1] == 1:
            return
        else:
            nrow = floor(sqrt((nshape[1] + 1)))
            ncol = ceil((nshape[1] + 1) / nrow)
            # error plots
            fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 3.2))
            axs = axs.flatten()
            for iatom in range(nshape[1]):
                xfeature = []
                truecomp = []
                predcomp = []
                for isamp in range(nshape[0]):
                    xfeature.append(x_atomfeature[isamp][iatom])
                    truecomp.append(self.true_values[isamp][iatom])
                    predcomp.append(self.predicted_values[isamp][iatom])
                hist1d, bin_edges = np.histogram(
                    np.array(predcomp) - np.array(truecomp), bins=40, density=True
                )
                ax = axs[iatom]
                ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
                ax.set_title("atom:" + str(iatom))

            ax = axs[nshape[1]]
            xfeature = []
            truecomp = []
            predcomp = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(x_atomfeature[isamp][:]))
                truecomp.append(sum(self.true_values[isamp][:]))
                predcomp.append(sum(self.predicted_values[isamp][:]))
            hist1d, bin_edges = np.histogram(
                np.array(predcomp) - np.array(truecomp), bins=40, density=True
            )
            ax.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            ax.set_title("SUM")

            ax = axs[nshape[1] + 1]  # summation over all the samples for each atom/node
            xfeature = []
            truecomp = []
            predcomp = []
            for iatom in range(nshape[1]):
                xfeature.append(sum(x_atomfeature[:][iatom]))
                truecomp.append(sum(self.true_values[:][iatom]))
                predcomp.append(sum(self.predicted_values[:][iatom]))
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

    def create_scatter_plot_atoms_vec(
        self, varname, x_atomfeature, iepoch=None, save_plot=True
    ):
        """Creates scatter plots for scalar output and vector outputs."""

        nshape = np.asarray(self.predicted_values).shape
        predicted_vec = np.reshape(
            np.asarray(self.predicted_values), (nshape[0], -1, 3)
        )
        true_vec = np.reshape(np.asarray(self.true_values), (nshape[0], -1, 3))
        num_nodes = true_vec.shape[1]

        markers_vec = ["o", "s", "d"]  # different markers for three vector components
        nrow = floor(sqrt((num_nodes + 1)))
        ncol = ceil((num_nodes + 1) / nrow)
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
        axs = axs.flatten()
        for iatom in range(num_nodes):
            for icomp in range(3):
                xfeature = []
                truecomp = []
                predcomp = []
                for isamp in range(nshape[0]):
                    xfeature.append(x_atomfeature[isamp][iatom])
                    truecomp.append(true_vec[isamp, iatom, icomp])
                    predcomp.append(predicted_vec[isamp, iatom, icomp])
                ax = axs[iatom]
                self.__scatter_impl(
                    ax,
                    truecomp,
                    predcomp,
                    s=6,
                    c=xfeature,
                    marker=markers_vec[icomp],
                    title="atom:" + str(iatom),
                    xylim_equal=True,
                )

        ax = axs[num_nodes]  # summation over all the atoms/nodes
        for icomp in range(3):
            xfeature = []
            truecomp = []
            predcomp = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(x_atomfeature[isamp][:]))
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

        ax = axs[num_nodes + 1]  # summation over all the samples for each atom/node
        for icomp in range(3):
            xfeature = []
            truecomp = []
            predcomp = []
            for iatom in range(num_nodes):
                xfeature.append(sum(x_atomfeature[:][iatom]))
                truecomp.append(sum(true_vec[:, iatom, icomp]))
                predcomp.append(sum(predicted_vec[:, iatom, icomp]))
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

    def plot_history(
        self,
        trainlib,
        vallib,
        testlib,
        tasklib,
        tasklib_vali,
        tasklib_test,
        tasklib_nodes,
        tasklib_vali_nodes,
        tasklib_test_nodes,
        task_weights,
        task_names,
    ):
        nrow = 1
        fhist = open(f"./logs/{self.model_with_config_name}/history_loss.pckl", "wb")
        pickle.dump(
            [
                trainlib,
                vallib,
                testlib,
                tasklib,
                tasklib_vali,
                tasklib_test,
                tasklib_nodes,
                tasklib_vali_nodes,
                tasklib_test_nodes,
                task_weights,
                task_names,
            ],
            fhist,
        )
        fhist.close()
        num_tasks = len(tasklib[0])
        if num_tasks > 0:
            tasklib = np.array(tasklib)
            tasklib_vali = np.array(tasklib_vali)
            tasklib_test = np.array(tasklib_test)
            nrow = 2
        fig, axs = plt.subplots(nrow, num_tasks, figsize=(16, 6 * nrow))
        axs = axs.flatten()
        ax = axs[0]
        ax.plot(trainlib, "-", label="train")
        ax.plot(vallib, ":", label="validation")
        ax.plot(testlib, "--", label="test")
        ax.set_title("total loss")
        ax.set_xlabel("Epochs")
        ax.set_yscale("log")
        ax.legend()
        for iext in range(1, num_tasks):
            axs[iext].axis("off")
        for ivar in range(tasklib.shape[1]):
            ax = axs[num_tasks + ivar]
            ax.plot(tasklib[:, ivar], label="train")
            ax.plot(tasklib_vali[:, ivar], label="validation")
            ax.plot(tasklib_test[:, ivar], "--", label="test")
            ax.set_title(task_names[ivar] + ", {:.4f}".format(task_weights[ivar]))
            ax.set_xlabel("Epochs")
            ax.set_yscale("log")
            if ivar == 0:
                ax.legend()
        for iext in range(num_tasks + tasklib.shape[1], axs.size):
            axs[iext].axis("off")
        plt.subplots_adjust(
            left=0.1, bottom=0.08, right=0.98, top=0.9, wspace=0.25, hspace=0.3
        )
        fig.savefig(f"./logs/{self.model_with_config_name}/history_loss.png")
        plt.close()
