import adios2 as ad2
import numpy as np
import pickle
import os
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font = {"size": 12}
matplotlib.rc("font", **font)


def histplot(dataset_list):
    for dataname in dataname_list:
        x = np.concatenate(dataset_list[dataname])
        if len(x) > 0:
            # print(dataname, x.min(), x.max(), x.mean(), x.std())
            h, bins = np.histogram(x, bins=50)
            plt.figure(figsize=[6, 3])
            plt.hist(x, bins=50, density=True, log=True)
            plt.title(dataname)
            plt.close()
        else:
            print(dataname, "no data")


def histplot2(dataset_list, name):
    datasetname = ["trainset", "valset", "testset"]
    for dataname in tqdm(dataname_list, desc="hist2"):
        fname = f"hist-3set-{name}-h-{dataname}.npz"
        if not os.path.exists(fname):
            xa = np.concatenate(dataset_list[dataname])
            h, bins = np.histogram(xa, bins=50)
            np.savez(fname, h=h, bins=bins)
        else:
            with np.load(fname) as f:
                h = f["h"]
                bins = f["bins"]

        plt.figure(figsize=[6, 3])
        for i in range(3):
            x = dataset_list[dataname][i]
            if len(x) > 0:
                # print(dataname, x.min(), x.max(), x.mean(), x.std())
                fname = f"hist-3set-{name}-h-{dataname}-{i}.npz"
                if not os.path.exists(fname):
                    h, _ = np.histogram(x, bins=bins, density=True)
                    np.savez(fname, h=h)
                else:
                    with np.load(fname) as f:
                        h = f["h"]
                plt.bar(
                    0.5 * bins[:-1] + 0.5 * bins[1:],
                    h,
                    width=bins[1] - bins[0],
                    alpha=0.2,
                    label="_Hidden",
                )
                # h, _, _ = plt.hist(x, bins=bins, alpha=0.2, label="_Hidden")
                xs = list()
                ys = list()
                xs.append(bins[0])
                ys.append(0)
                for k in range(len(h)):
                    xs.append(bins[k])
                    xs.append(bins[k + 1])
                    ys.append(h[k])
                    ys.append(h[k])
                xs.append(bins[-1])
                ys.append(0)
                plt.plot(xs, ys, label=datasetname[i])
            else:
                print(dataname, "no data")
        plt.yscale("log")
        plt.title(dataname.replace("-v2", ""))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"hist_3set-{name}-{dataname}.pdf")
        plt.close()


def histplot3(dataset_list, name):
    datasetname = ["trainset", "valset", "testset"]
    fig, ax = plt.subplots(1, 5, sharey=True, figsize=[16, 3])
    for p, dataname in tqdm(
        enumerate(dataname_list), desc="hist3", total=len(dataname_list)
    ):
        fname = f"hist-3set-{name}-h-{dataname}.npz"
        if not os.path.exists(fname):
            xa = np.concatenate(dataset_list[dataname])
            h, bins = np.histogram(xa, bins=50)
            np.savez(fname, h=h, bins=bins)
        else:
            with np.load(fname) as f:
                h = f["h"]
                bins = f["bins"]

        for i in range(3):
            x = dataset_list[dataname][i]
            if len(x) > 0:
                # print(dataname, x.min(), x.max(), x.mean(), x.std())
                fname = f"hist-3set-{name}-h-{dataname}-{i}.npz"
                if not os.path.exists(fname):
                    h, _ = np.histogram(x, bins=bins, density=True)
                    np.savez(fname, h=h)
                else:
                    with np.load(fname) as f:
                        h = f["h"]
                ax[p].bar(
                    0.5 * bins[:-1] + 0.5 * bins[1:],
                    h,
                    width=bins[1] - bins[0],
                    alpha=0.2,
                    label="_Hidden",
                )
                xs = list()
                ys = list()
                xs.append(bins[0])
                ys.append(0)
                for k in range(len(h)):
                    xs.append(bins[k])
                    xs.append(bins[k + 1])
                    ys.append(h[k])
                    ys.append(h[k])
                xs.append(bins[-1])
                ys.append(0)
                ax[p].plot(xs, ys, label=datasetname[i])
            else:
                print(dataname, "no data")
        ax[p].set_yscale("log")
        ax[p].set_title(dataname.replace("-v2", ""))
        ax[p].tick_params(axis="x", labelrotation=30)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.legend(loc=1, prop={"size": 10})
    plt.savefig(f"hist_3set-{name}-all.pdf")
    plt.close()


if __name__ == "__main__":
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.join(dirpwd, "dataset")
    # dataname_list = ["ANI1x", "MPTrj", "qm7x", "OC2022", "OC2020", "OC2020-20M"]
    # dataname_list = ["ANI1x-v2", "MPTrj-v2", "qm7x-v2", "OC2022-v2", "OC2020-v2", "OC2020-20M-v2"]
    dataname_list = ["ANI1x-v2", "MPTrj-v2", "qm7x-v2", "OC2022-v2", "OC2020-v2"]
    suffix = "-v2"

    ## atoms
    natom_list = dict()
    for dataname in tqdm(dataname_list, desc="atom"):
        natom_list[dataname] = list()
        for label in ["trainset", "valset", "testset"]:
            with ad2.open(os.path.join(prefix, dataname + ".bp"), "r") as f:
                f.__next__()
                natom = f.read(f"{label}/pos/variable_count")
                natom_list[dataname].append(natom)

    for dataname in dataname_list:
        x = np.concatenate(natom_list[dataname])
        # print(dataname, x.min(), x.max(), x.mean(), x.std())
        h, bins = np.histogram(x, bins=50)
        plt.figure(figsize=[6, 3])
        plt.hist(x, bins=50, density=True, log=True)
        plt.title(dataname)
        plt.close()

    plt.figure(figsize=[6, 3])
    for dataname in dataname_list:
        x = np.concatenate(natom_list[dataname])
        h, bins = np.histogram(x, bins=50, density=True)
        plt.plot(
            bins[:-1], h * (bins[1] - bins[0]) * 100, label=dataname.replace("-v2", "")
        )
        plt.fill_between(
            bins[:-1], h * (bins[1] - bins[0]) * 100, alpha=0.5, label="_nolegend_"
        )
        plt.xlabel("Num. of atoms")
        plt.ylabel("Ratio (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hist-atoms{suffix}.pdf")
    plt.close()

    ## edges
    edge_list = dict()
    for dataname in tqdm(dataname_list, desc="edge"):
        edge_list[dataname] = list()
        for label in ["trainset", "valset", "testset"]:
            with ad2.open(os.path.join(prefix, dataname + ".bp"), "r") as f:
                f.__next__()
                nedge = f.read(f"{label}/edge_attr/variable_count")
                edge_list[dataname].append(nedge)

    for dataname in dataname_list:
        x = np.concatenate(edge_list[dataname])
        # print(dataname, x.min(), x.max(), x.mean(), x.std())
        h, bins = np.histogram(x, bins=50)
        plt.figure(figsize=[6, 3])
        plt.hist(x, bins=50, density=True, log=True)
        plt.title(dataname)
        plt.close()

    plt.figure(figsize=[6, 3])
    for dataname in dataname_list:
        x = np.concatenate(edge_list[dataname])
        h, bins = np.histogram(x, bins=50, density=True)
        plt.plot(
            bins[:-1], h * (bins[1] - bins[0]) * 100, label=dataname.replace("-v2", "")
        )
        plt.fill_between(
            bins[:-1], h * (bins[1] - bins[0]) * 100, alpha=0.5, label="_nolegend_"
        )
        plt.xlabel("Num. of edges")
        plt.ylabel("Ratio (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hist-edges{suffix}.pdf")
    plt.close()

    ## energy
    energy_list = dict()
    for dataname in tqdm(dataname_list, desc="energy"):
        energy_list[dataname] = list()
        for label in ["trainset", "valset", "testset"]:
            with ad2.open(os.path.join(prefix, dataname + ".bp"), "r") as f:
                f.__next__()
                energy = f.read(f"{label}/energy")
                energy_list[dataname].append(energy)

    for dataname in dataname_list:
        x = np.concatenate(energy_list[dataname])
        if len(x) > 0:
            # print(dataname, x.min(), x.max(), x.mean(), x.std())
            h, bins = np.histogram(x, bins=50)
            plt.figure(figsize=[6, 3])
            plt.hist(x, bins=50, density=True, log=True)
            plt.title(dataname)
            plt.close()
        else:
            print(dataname, "no data")

    plt.figure(figsize=[6, 3])
    min_list = list()
    max_list = list()
    for dataname in dataname_list:
        min_list.append(energy_list[dataname][0].min())
        min_list.append(energy_list[dataname][1].min())
        min_list.append(energy_list[dataname][2].min())
        max_list.append(energy_list[dataname][0].max())
        max_list.append(energy_list[dataname][1].max())
        max_list.append(energy_list[dataname][2].max())
    mn, mx = min(min_list), max(max_list)
    bins = np.arange(mn, mx, 0.2)

    for dataname in dataname_list:
        x = np.concatenate(energy_list[dataname])
        h, bins = np.histogram(x, bins=bins, density=True)
        plt.plot(
            bins[:-1], h * (bins[1] - bins[0]) * 100, label=dataname.replace("-v2", "")
        )
        plt.fill_between(
            bins[:-1], h * (bins[1] - bins[0]) * 100, alpha=0.5, label="_nolegend_"
        )
        plt.xlabel("Energy")
        plt.ylabel("Ratio (%)")
    plt.xlim([-1000, +10])
    plt.legend(loc="upper right", bbox_to_anchor=(0.95, 1.0))

    ax = plt.gca()
    ax = inset_axes(ax, width="40%", height="40%", loc="upper left")
    for dataname in dataname_list:
        x = np.concatenate(energy_list[dataname])
        h, bins = np.histogram(x, bins=bins, density=True)
        ax.plot(
            bins[:-1], h * (bins[1] - bins[0]) * 100, label=dataname.replace("-v2", "")
        )
        ax.fill_between(
            bins[:-1], h * (bins[1] - bins[0]) * 100, alpha=0.5, label="_nolegend_"
        )
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"hist-energy{suffix}.pdf")
    plt.close()

    ## force
    force_list = dict()
    for dataname in tqdm(dataname_list, desc="force"):
        force_list[dataname] = list()
        for label in ["trainset", "valset", "testset"]:
            with ad2.open(os.path.join(prefix, dataname + ".bp"), "r") as f:
                f.__next__()
                force = f.read(f"{label}/force")
                force_list[dataname].append(force)

    # for dataname in dataname_list:
    #     x = np.concatenate(force_list[dataname])
    #     x = np.linalg.norm(x, axis=-1)
    #     # print(dataname, x.min(), x.max(), x.mean(), x.std())
    #     h, bins = np.histogram(x, bins=50)
    #     plt.figure(figsize=[6, 3])
    #     plt.hist(x, bins=50, density=True, log=True)
    #     plt.title(dataname)
    #     plt.close()

    min_list = list()
    max_list = list()
    for dataname in dataname_list:
        min_list.append(force_list[dataname][0].min())
        min_list.append(force_list[dataname][1].min())
        min_list.append(force_list[dataname][2].min())
        max_list.append(force_list[dataname][0].max())
        max_list.append(force_list[dataname][1].max())
        max_list.append(force_list[dataname][2].max())
    mn, mx = min(min_list), max(max_list)
    bins = np.arange(mn, mx, 0.2)

    h_list = dict()
    for dataname in tqdm(dataname_list, desc="hist"):
        fname = f"hist-h-{dataname}.npz"
        if not os.path.exists(fname):
            x = np.concatenate(force_list[dataname])
            x = np.linalg.norm(x, axis=-1)
            h, bins = np.histogram(x, bins=bins, density=True)
            np.savez(f"hist-h-{dataname}.npz", h=h)
            h_list[dataname] = h
        else:
            with np.load(fname) as f:
                h = np.load(fname)["h"]
            h_list[dataname] = h

    plt.figure(figsize=[6, 3])
    for dataname in dataname_list:
        h = h_list[dataname]
        plt.plot(
            bins[:-1], h * (bins[1] - bins[0]) * 100, label=dataname.replace("-v2", "")
        )
        plt.fill_between(
            bins[:-1], h * (bins[1] - bins[0]) * 100, alpha=0.5, label="_nolegend_"
        )
        plt.xlabel("Force")
        plt.ylabel("Ratio (%)")
    plt.xlim([-0.5, 10])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.close()

    # Create an inset axis within the main plot
    ax = plt.gca()
    ax = inset_axes(ax, width="40%", height="40%", loc="upper center")
    for dataname in dataname_list:
        h = h_list[dataname]
        ax.plot(
            bins[:-1], h * (bins[1] - bins[0]) * 100, label=dataname.replace("-v2", "")
        )
        ax.fill_between(
            bins[:-1], h * (bins[1] - bins[0]) * 100, alpha=0.5, label="_nolegend_"
        )

    plt.savefig(f"hist-force{suffix}.pdf")
    plt.close()

    histplot2(energy_list, "energy")
    histplot2(force_list, "force")

    histplot3(energy_list, "energy")
    histplot3(force_list, "force")
