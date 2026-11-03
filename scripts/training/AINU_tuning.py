"""
Adapted by Davide Carnevali from:
https://github.com/pytorch/examples/blob/main/imagenet/main.py

BSD 3-Clause License

Copyright (c) 2017,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Last modified by Carlotta Viana and Misha 05/2025
removed the randomization of the dataset. Instead use a split dataset creaded with Split-dataset.py
also added argument "medmnsit" to use medmnsit pretraned weights

"""

import argparse
import os
import shutil
import time
import warnings
import cv2
import sys
import numpy as np
import pandas as pd

import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.utils.data import Dataset
import albumentations as A


from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pylab as plt
from more_itertools import collapse
from itertools import cycle
import seaborn as sns


torch.cuda.empty_cache()
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names.append("xception")


is_mps_available = hasattr(torch.backends, "mps")
if is_mps_available:
    if torch.backends.mps.is_available():
        is_mps_available = True
    else:
        is_mps_available = False


parser = argparse.ArgumentParser(description="PyTorch CNN Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument("-a",    "--arch",    metavar="ARCH",    default="resnet18",    choices=model_names,    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",)
parser.add_argument("-j",    "--workers",    default=4,    type=int,    metavar="N",    help="number of data loading workers (default: 4)",)
parser.add_argument("-classes", "--classes", type=int, help="Number of classes")
parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch",    default=0,    type=int,    metavar="N",    help="manual epoch number (useful on restarts)",)
parser.add_argument("-b",    "--batch-size",    default=256,    type=int,    metavar="N",    help="mini-batch size (default: 256), this is the total "    "batch size of all GPUs on the current node when "    "using Data Parallel or Distributed Data Parallel",)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--fraction", default=0.2, type=float, help="Fraction of samples to use for validation and testing")
parser.add_argument("--wd",    "--weight-decay",    default=1e-4,    type=float,    metavar="W",    help="weight decay (default: 1e-4)",    dest="weight_decay",)
parser.add_argument("-p",    "--print-freq",    default=10,    type=int,    metavar="N",    help="print frequency (default: 10)",)
parser.add_argument("--resume",    default="",    type=str,    metavar="PATH",    help="path to latest checkpoint (default: none)",)
parser.add_argument("-e",    "--evaluate",    dest="evaluate",    action="store_true",    help="evaluate model on validation set",)
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("-o", "--outdir", help="Output directory")
parser.add_argument("-n", "--name", help="Output name")# Variable related to splits of the dataset
parser.add_argument("-display_labels",    "--display_labels",    help="name of sample split",)
parser.add_argument("-sname",    "--samplesname",    help="name of sample split",)
parser.add_argument("-test_only_celltypes",    "--test_only_celltypes",    nargs="+",    help="List of celltypes to keep only in the testing set",)
parser.add_argument("-train_val_only_celltypes",    "--train_val_only_celltypes",    nargs="+",    help="List of celltypes to keep only in the training set",)
parser.add_argument("--rewrite",    dest="rewrite",    action="store_true",    help="rewrite the column in name of sample split",)
parser.add_argument("-plot_pie",    "--plot_pie",    dest="plot_pie",    action="store_true",    help="plot pie chart of split",)
parser.add_argument("-augdir", "--augdir", help="Path to Albumentation pipeline")
parser.add_argument("-aug", "--augment", help="Augmentation pipeline")
parser.add_argument("-L", "--filelist", help="File containing details on samples and conditions")
parser.add_argument("-opt",    "--optimizer",    choices=["SGD", "Adam", "AdamW", "Adamax", "Adadelta", "Adagrad", "RMSprop"],    help="Optimizer",)
parser.add_argument("-medmnist", type=bool, default=False, help="True if use medmnist weights")
parser.add_argument("-mdir", help="medmnist model dir")
parser.add_argument("-model", help="medmnist model dir")
parser.add_argument("--loss_function",    type=str,    default="CrossEntropyLoss",    choices=[        "CrossEntropyLoss",        "MSELoss",        "BCEWithLogitsLoss",        "L1Loss",        "SmoothL1Loss",    ],    help="Loss function to use for training",)
parser.add_argument("--scheduler",    type=str,    default="one_cycle_lr",    choices=[        "step_lr",        "multi_step_lr",        "cyclic_lr",        "cosine_annealing",        "reduce_on_plateau",        "one_cycle_lr",        "lambda_lr",        "multiplicative_lr",        "exponential_lr",    ],    help="Type of learning rate scheduler",)
parser.add_argument("--pie_cmap", help="Color map for pie chart")
parser.add_argument("--only_test", dest="only_test", action="store_true", help="Only test the model")
parser.add_argument("--test_model_path", help="Path to the model to test")
parser.add_argument("--test_data_path", help="File containing details on samples and conditions")
parser.add_argument("--test_images_dir", help="Folder containing test images")
# Scheduler-specific arguments
parser.add_argument("--step_size", type=int, default=30, help="Step size for StepLR/MultiStepLR")
parser.add_argument("--gamma",    type=float,    default=1.0,    help="Gamma for StepLR/ExponentialLR/ReduceLROnPlateau",)
parser.add_argument("--milestones",    nargs="+",    type=int,    default=[30, 60],    help="Milestones for MultiStepLR",)
parser.add_argument("--warmup", type=int, default=0, help="Linear warmup step size")
# LambdaLR and MultiplicativeLR specific arguments
parser.add_argument("--lr_lambda",    type=str,    default="lambda epoch: 0.95 ** epoch",    help="Lambda function for LambdaLR and MultiplicativeLR",)# CyclicLR specific arguments
parser.add_argument("--lr",    "--learning-rate",    default=0.001,    type=float,    metavar="LR",    help="initial learning rate",    dest="lr",)
parser.add_argument("-oclr",    "--ocmaxlr",    type=float,    default=0.005,    help="Max learning rate for CyclicLR and OneCycleLR",)
parser.add_argument("--step_size_up", type=int, default=2000, help="Step size up for CyclicLR")
# Cosine Annealing specific arguments
parser.add_argument("--T_max", type=int, default=50, help="T_max for CosineAnnealingLR")
parser.add_argument("--eta_min",    type=float,    default=0,    help="Minimum learning rate for CosineAnnealingLR",)
# ReduceLROnPlateau specific arguments
parser.add_argument("--patience", type=int, default=10, help="Patience for ReduceLROnPlateau")

args = parser.parse_args()


logfile = args.outdir + args.name + ".log"
best_acc1 = 0


def parse_display_labels(display_labels):
    return {x.split("=")[0]: x.split("=")[1] for x in display_labels.split(",")}


class ImageLoadError(Exception):
    """Custom exception for image loading errors."""

    pass


def get_scheduler(optimizer, scheduler_name, **kwargs):
    if scheduler_name == "step_lr":
        return lr_scheduler.StepLR(
            optimizer, step_size=kwargs["step_size"], gamma=kwargs["gamma"]
        )
    elif scheduler_name == "multi_step_lr":
        return lr_scheduler.MultiStepLR(
            optimizer, milestones=kwargs["milestones"], gamma=kwargs["gamma"]
        )
    elif scheduler_name == "cyclic_lr":
        return lr_scheduler.CyclicLR(
            optimizer,
            base_lr=kwargs["base_lr"],
            max_lr=kwargs["max_lr"],
            step_size_up=kwargs["step_size_up"],
            mode="triangular",
        )
    elif scheduler_name == "cosine_annealing":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs["T_max"], eta_min=kwargs["eta_min"]
        )
    elif scheduler_name == "reduce_on_plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=kwargs["gamma"], patience=kwargs["patience"]
        )
    elif scheduler_name == "one_cycle_lr":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs["max_lr"],
            steps_per_epoch=kwargs["steps_per_epoch"],
            epochs=kwargs["epochs"],
            cycle_momentum=True
        )
    elif scheduler_name == "lambda_lr":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=kwargs["lr_lambda"])
    elif scheduler_name == "multiplicative_lr":
        return lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=kwargs["lr_lambda"])
    elif scheduler_name == "exponential_lr":
        return lr_scheduler.ExponentialLR(optimizer, gamma=kwargs["gamma"])
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def split_dataset(
    lista,
    split_name,
    plot_pie,
    rewrite,
    test_only_celltypes=[],
    train_val_only_celltypes=[],
):
    """
    Splits the dataset into training, validation, and test sets,
    ensuring some celltypes are exclusively in the training set if specified.

    Parameters:
    - lista: DataFrame containing dataset information.
    - split_name: Name of the column to store the split information.
    - plot_pie: If True, plot a pie chart of the distribution.
    - rewrite: If True, overwrite an existing split column.
    - test_only_celltypes: List of celltypes to keep only in the training set.
    - train_val_only_celltypes: List of celltypes to keep only in the training set.
    """
    lista_split = lista.copy()
    if split_name in lista_split.columns and not rewrite:
        print(f"{split_name} already exists, we use the existing split")
    else:
        if test_only_celltypes:
            test_only_df = lista_split[
                lista_split["celltype"].isin(test_only_celltypes)
            ]
            lista = lista_split[~lista_split["celltype"].isin(test_only_celltypes)]
            lista_split.loc[test_only_df.index, split_name] = "test"
        else:
            test_only_df = None

        # TEST
        label_names = sorted(lista.new_labels.unique())
        for x in label_names:
            tmp = []
            celltypes = lista[lista["new_labels"] == x]["celltype"].unique()
            for celltype in celltypes:
                if train_val_only_celltypes:
                    if celltype in train_val_only_celltypes:
                        continue
                tmp.extend(
                    lista[(lista["celltype"] == celltype) & (lista["new_labels"] == x)]
                    .sample(frac=args.fraction)["File"]
                    .values
                )
            lista = lista[~lista["File"].isin(tmp)]
            lista_split.loc[lista_split["File"].isin(tmp), split_name] = "test"
        # VAL
        for x in label_names:
            tmp = []
            celltype = lista[lista["new_labels"] == x]["celltype"].unique()
            for cell in celltype:
                tmp.extend(
                    lista[(lista["celltype"] == cell) & (lista["new_labels"] == x)]
                    .sample(frac=args.fraction)["File"]
                    .values
                )
            lista = lista[~lista["File"].isin(tmp)]
            lista_split.loc[lista_split["File"].isin(tmp), split_name] = "val"
        # TRAIN
        train_images = lista["File"].values.tolist()
        lista_split.loc[lista_split["File"].isin(train_images), split_name] = "train"

        if test_only_df is not None:
            lista_split.loc[test_only_df.index, split_name] = "test"
        assert len(lista_split[lista_split[split_name] == "train"]) > 0, (
            "No images in the training set"
        )
        assert len(lista_split[lista_split[split_name] == "val"]) > 0, (
            "No images in the validation set"
        )
        assert len(lista_split[lista_split[split_name] == "test"]) > 0, (
            "No images in the test set"
        )
        print(
            f"Splitting dataset into training, validation, and test sets based on {split_name} column"
        )
    # Plot pie chart based on the split

    if plot_pie:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        def create_gradient(color1, color2, steps=100):
            """
            Create a gradient between two colors.

            Parameters:
            - color1: Tuple of RGB values (e.g., (1, 0, 0) for red).
            - color2: Tuple of RGB values (e.g., (0, 0, 1) for blue).
            - steps: Number of intermediate colors in the gradient.

            Returns:
            - A numpy array representing the gradient.
            """
            gradient = np.zeros((steps + 2, 3))
            for i in range(3):  # Interpolating R, G, B channels
                gradient[:, i] = np.linspace(color1[i], color2[i], steps + 2)
            return gradient[1 : len(gradient) - 1, :]

        def get_distinct_colors(colormap_name="viridis", num_colors=5):
            cmap = plt.colormaps[colormap_name]
            range1 = int(len(cmap.colors) / (num_colors + 1))
            colors = [cmap(i * range1) for i in range(num_colors)]
            return colors

        grouped_condition = (
            lista_split.groupby([split_name, "new_labels"]).count().reset_index()
        )
        grouped_condition.groupby(split_name)["File"].apply(list).to_list()
        grouped_df = lista_split.groupby(
            [split_name, "new_labels", "celltype"]
        ).count()["File"]
        # Prepare data for each level
        inner_counts = grouped_df.groupby(
            [split_name, "new_labels"]
        ).sum()  # First level
        middle_counts = grouped_df.groupby(
            [split_name, "new_labels", "celltype"]
        ).sum()  # Second level
        outer_counts = grouped_df.groupby([split_name]).sum()

        inner_label = [item[1] for item in inner_counts.index]
        middle_label = [item[2] for item in middle_counts.index]
        outer_label = outer_counts.index

        # cmap = plt.colormaps["tab20b"]
        if args.pie_cmap is None:
            outer_colors = [
                (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
                (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
                (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
            ]
        else:
            outer_colors = get_distinct_colors(
                args.pie_cmap, num_colors=3
            )  # cmap(np.arange(3)*4)

        inner_colors = []
        for i in range(len(outer_colors)):
            inner_colors.extend(
                create_gradient(
                    outer_colors[i],
                    (1, 1, 1),
                    steps=len(inner_counts.loc[outer_label[i]]),
                )
            )
        middle_colors = []
        for i in range(len(outer_colors)):
            middle_colors.extend(
                create_gradient(
                    outer_colors[i],
                    (1, 1, 1),
                    steps=len(middle_counts.loc[outer_label[i]]),
                )
            )

        startangle = 0
        # Plot
        fig, ax = plt.subplots(figsize=(8, 9))

        # Outer Ring
        wedges, texts = ax.pie(
            outer_counts,
            radius=1.3,
            colors=outer_colors,
            wedgeprops=dict(width=0.4, edgecolor="w"),
            startangle=startangle,
            textprops={"fontsize": 14},
        )

        # Middle Ring
        ax.pie(
            middle_counts,
            radius=1.0,
            colors=middle_colors,
            labels=middle_label,
            wedgeprops=dict(width=0.4, edgecolor="w"),
            labeldistance=0.67,
            startangle=startangle,
            textprops={"fontsize": 12},
        )

        # Inner Ring
        ax.pie(
            inner_counts,
            radius=0.6,
            colors=inner_colors,
            labels=inner_label,
            wedgeprops=dict(width=0.4, edgecolor="w"),
            pctdistance=0.5,
            labeldistance=0.45,
            startangle=startangle,
            textprops={"fontsize": 14},
        )

        bbox_props = dict(boxstyle="round,pad=0.4", fc="w", ec="k", lw=0.3)
        kw = dict(
            arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center"
        )

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(
                outer_counts.index[i] + "\n" + str(outer_counts[i]) + " cells",
                xy=(x, y),
                fontsize=11,
                xytext=(1.4 * np.sign(x), 1.7 * y),
                horizontalalignment=horizontalalignment,
                **kw,
            )
        if test_only_celltypes:
            title = f"Dataset of {outer_counts.sum()} images ({args.samplesname})\n Keep for testing only: {', '.join(test_only_celltypes)}"
            if train_val_only_celltypes:
                title += f" and \n for train/val: {', '.join(train_val_only_celltypes)}"
        elif train_val_only_celltypes and not test_only_celltypes:
            title = f"Dataset of {outer_counts.sum()} images ({args.samplesname})\n Keep for training/val only: {', '.join(train_val_only_celltypes)}"
        else:
            title = f"Dataset of {outer_counts.sum()} images ({args.samplesname})"
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        os.makedirs(args.outdir, exist_ok=True)
        plt.savefig(
            f"{args.outdir}/{args.name}_{args.samplesname}_pie_chart.pdf",
            dpi=300,
            facecolor="white",
        )
    #### Plot pie without splitting into training, testing and validation
        # cmap = plt.colormaps["tab20b"]
        if args.pie_cmap is None:
            #inner_colors = [(0.37254901960784315, 0.29411764705882354, 0.5450980392156862, 1), (0.6784313725490196,0.9372549019607843,0.8196078431372549, 1)]
            inner_colors = get_distinct_colors(
                "tab10", num_colors=2
            )
        else:
            inner_colors = get_distinct_colors(
                args.pie_cmap, num_colors=2
            )  # cmap(np.arange(3)*4)

        grouped_condition = (
            lista_split.groupby(["new_labels"]).count().reset_index()
        )
        grouped_condition.groupby("new_labels")["File"].apply(list).to_list()
        grouped_df = lista_split.groupby(
            ["new_labels", "celltype"]
        ).count()["File"]
        # Prepare data for each level
        inner_counts = grouped_df.groupby(
            ["new_labels"]
        ).sum()  # First level
        middle_counts = grouped_df.groupby(
            ["new_labels", "celltype"]
        ).sum()  # Second level
        print("grouped_df", grouped_df)
        print("inner_counts", inner_counts)
        print("middle_counts", middle_counts)
        inner_label = [item for item in inner_counts.index]
        middle_label = [item[1] for item in middle_counts.index]
        print("inner_label", inner_label)
        print("middle_label", middle_label)
        # cmap = plt.colormaps["tab20b"]
        middle_colors = []
        print(middle_counts)
        for i in range(len(inner_colors)):
            middle_colors.extend(
                create_gradient(
                    inner_colors[i],
                    (1, 1, 1),
                    steps=len(middle_counts.loc[inner_label[i]]),
                )
            )

        startangle = 0
        # Plot
        fig, ax = plt.subplots(figsize=(8, 9))
        wedges, texts = ax.pie(
            middle_counts,
            radius=1.0,
            colors=middle_colors,
            #labels=middle_label,
            wedgeprops=dict(width=0.4 , edgecolor="w"),
            startangle=startangle,
            textprops={"fontsize": 14},
        )

        # Inner Ring
        ax.pie(
            inner_counts,
            radius=0.6,
            colors=inner_colors,
            labels=inner_label,
            wedgeprops=dict(width=0.4, edgecolor="w"),
            pctdistance=0.5,
            labeldistance=0.45,
            startangle=startangle,
            textprops={"fontsize": 14},
        )

        bbox_props = dict(boxstyle="round,pad=0.4", fc="w", ec="k", lw=0.3)
        kw = dict(
            arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center"
        )

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(
                middle_label[i] + "\n" + str(middle_counts[i]) + " cells",
                xy=(x, y),
                fontsize=11,
                xytext=(1.4 * np.sign(x), 1.7 * y),
                horizontalalignment=horizontalalignment,
                **kw,
            )

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f"{args.outdir}/{args.name}_{args.samplesname}_pie_chart_wo_split.pdf",
            dpi=300,
            facecolor="white",
        )

    return lista_split


def main():
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    train_acc1_list = []
    train_acc5_list = []
    train_loss_list = []
    val_acc1_list = []
    val_acc5_list = []
    val_loss_list = []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if is_mps_available else "cpu"
    )
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        if args.arch == "inception_v3":
            model = models.__dict__[args.arch](pretrained=True, aux_logits=False)
        elif args.arch == "xception":
            import timm

            model = timm.create_model("xception", pretrained=True)
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)

    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    ## MEDMNIST MODIFICATIONS BY CARLOTTA - March 2024

    # If medmnist is specified, used its weights to start the training
    if args.medmnist:
        medmnist_dir = args.mdir
        # load the checkpoints of medmnist
        # checkpoint_path = medmnist_dir
        checkpoint = torch.load(medmnist_dir + args.model + ".pth", map_location=device)
        # Get the key under which the state dictionary is stored
        state_dict_key = next(iter(checkpoint.keys()))
        # Extract the state dictionary from the loaded checkpoint
        state_dict = checkpoint[state_dict_key]
        # Extract the keys of the state dictionary from the chosen model
        ### mymodel_keys = model.state_dict().keys()
        medmnist_model_keys = state_dict.keys()

        # Replace "shortcut" with "downsample" and "linear" with "fc"
        renamed_medmnist_model_keys = [
            key.replace("shortcut", "downsample").replace("linear", "fc")
            for key in medmnist_model_keys
        ]

        # print("NEW: ", renamed_medmnist_model_keys)
        # print("MODEL: ", mymodel_keys)

        # Create a new dictionary with modified keys
        new_state_dict = {
            new_key: state_dict[old_key]
            for new_key, old_key in zip(renamed_medmnist_model_keys, state_dict.keys())
        }

        # change file name and save
        # checkpoint_filename = os.path.basename(checkpoint_path)
        # filename_without_extension = os.path.splitext(checkpoint_filename)[0]
        modified_filename = args.model + "_modified.pth"
        modified_checkpoint_path = os.path.join(medmnist_dir, modified_filename)
        torch.save({"state_dict": new_state_dict}, modified_checkpoint_path)

        # Load the modified state_dict into the model
        model.load_state_dict(
            torch.load(modified_checkpoint_path)["state_dict"], strict=False
        )
        print("new checkpoints loaded")

    # Reshape the final Layer to match the number of classes

    if args.arch == "resnet152":
        model.fc = nn.Linear(model.fc.in_features, args.classes)

    elif args.arch == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, args.classes)

    elif args.arch == "wide_resnet101_2":
        model.fc = nn.Linear(model.fc.in_features, args.classes)

    elif args.arch == "resnext101_32x8d":
        model.fc = nn.Linear(model.fc.in_features, args.classes)

    elif args.arch == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, args.classes)

    elif args.arch == "vgg16":
        model.classifier[6] = nn.Linear(4096, args.classes)

    elif args.arch == "vgg19_bn":
        model.classifier[6] = nn.Linear(4096, args.classes)

    elif args.arch == "squeezenet":
        model.classifier[1] = nn.Conv2d(
            512, args.classes, kernel_size=(1, 1), stride=(1, 1)
        )
    elif args.arch == "densenet121":
        model.classifier = nn.Linear(1024, args.classes)

    elif args.arch == "densenet201":
        model.classifier = nn.Linear(1920, args.classes)

    elif args.arch == "densenet161":
        model.classifier = nn.Linear(2208, args.classes)

    elif args.arch == "efficientnet_b7":
        model.classifier = nn.Linear(2560, args.classes)

    elif args.arch == "efficientnet_b3":
        model.classifier = nn.Linear(1536, args.classes)

    elif args.arch == "inception_v3":
        # model.AuxLogits.fc = nn.Linear(768, args.classes)
        model.fc = nn.Linear(2048, args.classes)

    elif args.arch == "googlenet":
        model.fc = nn.Linear(1024, args.classes)

    elif args.arch == "xception":
        model.fc = nn.Linear(2048, args.classes)

    elif args.arch == "alexnet":
        model.classifier[6] = nn.Linear(4096, args.classes)

    else:
        print("No such model...Sayonara!")
        exit()

    if torch.cuda.is_available():
        if args.gpu:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            device = torch.device("cuda:{}".format(args.gpu))
        else:
            device = torch.device("cuda")
            if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
                model.features = torch.nn.DataParallel(model.features)
    elif is_mps_available:
        args.workers = 0
        device = torch.device("mps")
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # define loss function (criterion) and optimizer
    if args.loss_function == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss_function == "MSELoss":
        criterion = nn.MSELoss().to(
            device
        )  # Mean Squared Error Loss (Regression tasks)
    elif args.loss_function == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().to(
            device
        )  # Binary Cross Entropy Loss for binary classification
    elif args.loss_function == "L1Loss":
        criterion = nn.L1Loss().to(
            device
        )  # L1 Loss (also called MAE) for regression with outliers
    elif args.loss_function == "SmoothL1Loss":
        criterion = nn.SmoothL1Loss().to(
            device
        )  # Less sensitive to outliers than MSELoss
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adamax":
        optimizer = torch.optim.SGD(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.SGD(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    else:
        print("No such optimizer implemented...")
        exit()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_dir = args.data

    lista = pd.read_csv(data_dir + args.filelist, sep="\t", header=0)
    display_labels = parse_display_labels(args.display_labels)
    print(lista.head())
    lista["new_labels"] = lista["condition"].map(display_labels)
    label_names = sorted(lista.condition.unique())
    labels = {}
    for i in range(len(label_names)):
        labels[display_labels[label_names[i]]] = i
    lista = split_dataset(
        lista,
        args.samplesname,
        plot_pie=args.plot_pie,
        rewrite=args.rewrite,
        test_only_celltypes=args.test_only_celltypes,
        train_val_only_celltypes=args.train_val_only_celltypes,
    )
    try:
        lista.to_csv(data_dir + args.filelist, sep="\t", header=True, index=False)
    except PermissionError:
        lista.to_csv(args.outdir + args.filelist, sep="\t", header=True, index=False)
    extension = os.path.splitext(os.listdir(data_dir)[0])[1]
    extension = extension if extension in (".jpg", ".png") else ".png"

    train_images_filepaths = [
        (data_dir + j["File"] + extension, labels[j["new_labels"]])
        for idx, j in lista[lista[args.samplesname] == "train"].iterrows()
    ]
    test_images_filepaths = [
        (data_dir + j["File"] + extension, labels[j["new_labels"]])
        for idx, j in lista[lista[args.samplesname] == "test"].iterrows()
    ]
    val_images_filepaths = [
        (data_dir + j["File"] + extension, labels[j["new_labels"]])
        for idx, j in lista[lista[args.samplesname] == "val"].iterrows()
    ]
    # print(train_images_filepaths)

    class MyDataset(Dataset):
        def __init__(self, images_filepaths, transform=None):
            self.data = [x[0] for x in images_filepaths]
            self.targets = [x[1] for x in images_filepaths]
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = cv2.imread(self.data[idx])
            # if image is None:
            #    raise ImageLoadError(
            #        f"Error: Unable to load image from {idx, self.data[idx]}"
            #    )
            # else:
            #    print("Image loaded successfully", self.data[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.targets[idx]
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image / 255, label

    train_transform = A.load(args.augdir + "train/" + args.augment + ".json")
    train_dataset = MyDataset(
        images_filepaths=train_images_filepaths, transform=train_transform
    )

    val_transform = A.load(args.augdir + "val/" + args.augment + ".json")
    val_dataset = MyDataset(
        images_filepaths=val_images_filepaths, transform=val_transform
    )

    test_transform = A.load(args.augdir + "test/" + args.augment + ".json")
    test_dataset = MyDataset(
        images_filepaths=test_images_filepaths, transform=test_transform
    )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.optimizer == "Adagrad" or args.optimizer == "Adadelta":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.ocmaxlr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            verbose=False,
            cycle_momentum=False,
        )
    else:
        if args.lr_lambda:
            lr_lambda = eval(args.lr_lambda)  # Convert string into a Python function

        scheduler = get_scheduler(
            optimizer,
            args.scheduler,
            step_size=args.step_size,
            gamma=args.gamma,
            milestones=args.milestones,
            base_lr=args.lr,
            max_lr=args.ocmaxlr,
            step_size_up=args.step_size_up,
            T_max=args.T_max,
            eta_min=args.eta_min,
            patience=args.patience,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            lr_lambda=lr_lambda
            if args.scheduler in ["lambda_lr", "multiplicative_lr"]
            else None,
        )

    lrs = []
    if not args.only_test:
        print("Training...")
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train_acc1, train_acc5, train_loss = train(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                device,
                args,
                scheduler,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            train_acc1_list.append(train_acc1.to("cpu").numpy())
            train_acc5_list.append(train_acc5.to("cpu").numpy())
            train_loss_list.append(train_loss)

            # scheduler.step() when using OneCycleLR the scheduler.step() is called
            # after each minibatch training

            # evaluate on validation set
            val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args)

            val_acc1_list.append(val_acc1.to("cpu").numpy())
            val_acc5_list.append(val_acc5.to("cpu").numpy())
            val_loss_list.append(val_loss)

            # remember best acc@1 and save checkpoint
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )

        # Setting plotting parameters
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams.update({"font.size": 14})
        output_plot_path = (
            args.outdir + args.name + "_" + str(args.start_epoch) + "_" + str(epoch)
        )
        plt.figure()
        plt.plot(lrs)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_plot_path + "_lr.pdf", dpi=300, facecolor="white")
        # Plot results
        plt.figure()
        plt.plot(train_loss_list, label="training loss")
        plt.plot(val_loss_list, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plot_path + "_loss.pdf", dpi=300, facecolor="white")

        plt.figure()
        plt.plot(train_acc1_list, label="Training accuracy")
        plt.plot(val_acc1_list, label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plot_path + "_accuracy.pdf", dpi=300, facecolor="white")
        sys.stdout = open(logfile, "w")
        print("Running parameters: {}".format(vars(args)))
        print(
            "Best Training Accuracy and Loss: {} / {}".format(
                max(train_acc1_list), min(train_loss_list)
            )
        )
        print(
            "Best Validation Accuracy and Loss: {} / {}".format(
                best_acc1, min(val_loss_list)
            )
        )
        print(
            "Training set: {}".format(
                lista[lista[args.samplesname] == "train"]["File"].to_list()
            )
        )
        print(
            "Validation set: {}".format(
                lista[lista[args.samplesname] == "val"]["File"].to_list()
            )
        )
        print(
            "Test set: {}".format(
                lista[lista[args.samplesname] == "test"]["File"].to_list()
            )
        )
    else:
        print("Only testing...")
        if args.test_data_path:
            lista_test = pd.read_csv(args.test_data_path, sep="\t", header=0)
            lista_test["new_labels"] = lista_test["condition"].map(display_labels)

            extension = os.path.splitext(os.listdir(args.test_images_dir)[0])[1]
            extension = extension if extension in (".jpg", ".png") else ".png"

            # sanity check for the test set
            # we check if the image names in the lista_test were not used in the training set or validation set in order to avoid data leakage
            # we need to compare column File with the list of images used in the training set and validation set
            # if the image is not in the list of images used for training and validation, we can use it for testing
            # return an error and list of problematic testing images that are in the list of images used for training and validation

            used_for_training = lista[lista[args.samplesname].isin(["train", "val"])][
                "File"
            ].to_list()
            problematic_testing = [
                x for x in lista_test["File"].to_list() if x in used_for_training
            ]
            if len(problematic_testing) > 0:
                print(
                    f"Error: The following {len(problematic_testing)} (out of {len(lista_test)}) testing images are in the training and validation set: ",
                    problematic_testing,
                )
                # save non problematic testing images to a file
                non_problematic_testing = lista_test[
                    ~lista_test["File"].isin(problematic_testing)
                ]
                name_test_file = os.path.basename(args.test_data_path)
                non_problematic_testing.to_csv(
                    args.outdir + "non_problematic_testing_images_" + name_test_file,
                    sep="\t",
                    header=True,
                    index=False,
                )
                print(
                    f"{len(non_problematic_testing)} non problematic testing images saved to {args.outdir}non_problematic_testing_images_{name_test_file}. Please check the file and rerun the script."
                )
                sys.exit(1)
            else:
                print("No problematic testing images found")
            test_images_filepaths = [
                (args.test_images_dir + j["File"] + extension, labels[j["new_labels"]])
                for idx, j in lista_test.iterrows()
            ]
            test_transform = A.load(args.augdir + "test/" + args.augment + ".json")
            test_dataset = MyDataset(
                images_filepaths=test_images_filepaths, transform=test_transform
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
            )
            sys.stdout = open(logfile, "a+")  # append mode
            print("Running parameters: {}".format(vars(args)))
            print("Test set: {}".format(lista_test["File"].to_list()))
        else:
            sys.stdout = open(logfile, "a+")
            print("Running parameters: {}".format(vars(args)))
            print(
                "Test set: {}".format(
                    lista[lista[args.samplesname] == "test"]["File"].to_list()
                )
            )

    # Challenge the model on real data test set
    print("Calculate Statistics on unseen test data...")
    if args.test_model_path:
        model.load_state_dict(
            torch.load(args.test_model_path, weights_only=True)["state_dict"]
        )
    else:
        model.load_state_dict(
            torch.load(
                args.outdir + args.name + "_model_best.pth.tar", weights_only=True
            )["state_dict"]
        )
    model.eval()
    sm = torch.nn.Softmax(dim=1)
    N_test = len(test_dataset)
    correct = 0
    y_pred_list = []
    real_labels = test_dataset.targets
    test_labels = []
    probabilities = []
    binary_prob_0 = []
    binary_prob_1 = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.float().to(device), y_test.to(device)
            z = model(x_test)
            probabilities.append(sm(z.to("cpu")).numpy())
            if args.classes == 2:
                binary_prob_0.append(sm(z.to("cpu"))[:, 0].numpy())
                binary_prob_1.append(sm(z.to("cpu"))[:, 1].numpy())
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
            y_pred_list.append(yhat.cpu().numpy())
            tmp = np.zeros((y_test.size()[0], args.classes))
            for i in range(len(y_test)):
                tmp[i][int(y_test[i])] = 1
            test_labels.append(tmp)

        accuracy = correct / N_test
        y_pred_list = list(collapse([a.squeeze().tolist() for a in y_pred_list]))

    print("Test Accuracy: {}".format(accuracy))
    probabilities = np.concatenate(probabilities)
    test_labels = np.concatenate(test_labels)
    if args.classes == 2:
        binary_prob_0 = np.concatenate(binary_prob_0).ravel()
        binary_prob_1 = np.concatenate(binary_prob_1).ravel()

    print(
        "F1 score : micro {}, macro {}, weighted {}  ".format(
            f1_score(real_labels, y_pred_list, average="micro"),
            f1_score(real_labels, y_pred_list, average="macro"),
            f1_score(real_labels, y_pred_list, average="weighted"),
        )
    )
    print(
        classification_report(
            real_labels, y_pred_list, labels = list(range(len(label_names))), target_names=label_names, digits=4
        )
    )

    if (args.classes > 2) & (len(set(real_labels))>2):
        print(
            "ROC AUC score: {}".format(
                roc_auc_score(
                    np.array(real_labels),
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
        )
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(args.classes):
            fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            test_labels.ravel(), probabilities.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 10
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(args.classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= args.classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(args.classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(
                    label_names[i], roc_auc[i]
                ),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right")
        plt.savefig(
            args.outdir + args.name + "_ROC_AUC.pdf", dpi=300, facecolor="white"
        )

    elif (args.classes==2 ) & (len(set(real_labels))==2):
        print(
            "ROC AUC score: {}".format(
                roc_auc_score(np.array(real_labels), np.array(binary_prob_1))
            )
        )
        RocCurveDisplay.from_predictions(real_labels, binary_prob_1, name="AINU")
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.savefig(
            args.outdir + args.name + "_ROC_AUC.pdf", dpi=300, facecolor="white"
        )

        PrecisionRecallDisplay.from_predictions(
            real_labels, binary_prob_0, pos_label=0, name="AINU"
        )
        plt.xlabel("Recall (Positive label: " + label_names[0] + ")")
        plt.ylabel("Precision (Positive label: " + label_names[0] + ")")
        plt.savefig(
            args.outdir + args.name + "_Precison_Recall_" + label_names[0] + ".pdf",
            dpi=300,
            facecolor="white",
        )

        PrecisionRecallDisplay.from_predictions(
            real_labels, binary_prob_1, pos_label=1, name="AINU"
        )
        plt.xlabel("Recall (Positive label: " + label_names[1] + ")")
        plt.ylabel("Precision (Positive label: " + label_names[1] + ")")
        plt.savefig(
            args.outdir + args.name + "_Precison_Recall_" + label_names[1] + ".pdf",
            dpi=300,
            facecolor="white",
        )
        plt.close()

    cm = confusion_matrix(real_labels, y_pred_list, labels = list(range(len(label_names))))
    norm_sum = cm.sum(axis=1)[:, np.newaxis]
    for j in range(len(norm_sum)):
        if norm_sum[j] == 0:
            norm_sum[j] = 1
    cm_normalized = cm.astype("float") / norm_sum

    ##if norm_sum==0:
    #    cm_normalized = cm.astype('float') /
    #    else cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    # Plot the non-normalized confusion matrix
    confusion_labels = np.array([f"{v_norm:.2f}" for v_norm in cm_normalized.flatten()])
    confusion_labels = confusion_labels.reshape(
        cm_normalized.shape[0], cm_normalized.shape[1]
    )

    sns.heatmap(
        cm_normalized,
        annot=confusion_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 28},
    )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Confusion Matrix with Normalized Values", fontsize=14)
    plt.ylabel("True Label", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.tight_layout()

    plt.savefig(
        args.outdir + args.name + "_ConfusionMatrix_normalized.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close()
    # Plot the non-normalized confusion matrix
    plt.figure(figsize=(8, 6))
    confusion_labels = np.array([f"{v}" for v in cm.flatten()])
    confusion_labels = confusion_labels.reshape(cm.shape[0], cm.shape[1])

    sns.heatmap(
        cm,
        annot=confusion_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 28},
    )
    plt.title("Confusion Matrix with Real Counts", fontsize=14)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.tight_layout()

    plt.savefig(
        args.outdir + args.name + "_ConfusionMatrix_notnorm.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close()

    # Create a figure and axis
    # Plot the non-normalized confusion matrix
    confusion_labels = np.array(
        [
            f"{v_norm:.2f}\n({int(v)})"
            for v, v_norm in zip(cm.flatten(), cm_normalized.flatten())
        ]
    )
    confusion_labels = confusion_labels.reshape(cm.shape[0], cm.shape[1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=confusion_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 28},
    )
    plt.title(
        "Confusion Matrix Normalized Values with Real Counts in Brackets", fontsize=14
    )
    plt.ylabel("True Label", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(
        args.outdir + args.name + "_ConfusionMatrix_notnorm+norm.pdf",
        dpi=300,
        facecolor="white",
    )
    plt.close()


def train(
    train_loader, model, criterion, optimizer, epoch, device, args, scheduler=None
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        train_acc1, train_acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(train_acc1[0], images.size(0))
        top5.update(train_acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return (top1.avg, top5.avg, losses.avg)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if is_mps_available else "cpu"
    )

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(device, non_blocking=True)
                target = target.cuda(device, non_blocking=True)
            if is_mps_available:
                images = images.to("mps", non_blocking=True)
                target = target.to("mps", non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return (top1.avg, top5.avg, losses.avg)


def save_checkpoint(
    state, is_best, filename=args.outdir + args.name + "_checkpoint.pth.tar"
):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.outdir + args.name + "_model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
