import logging
import os
import socket
import warnings
from pathlib import Path

import torch
import yaml
import pandas as pd
from torch_geometric.loader import DataLoader as GeoDataLoader

from point_vs import utils
from point_vs.global_objects import NUM_WORKERS
from point_vs.inference import get_model_and_test_dl

from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.parse_args import parse_args
from point_vs.preprocessing.data_loaders import get_data_loader
from point_vs.preprocessing.data_loaders import PygPointCloudDataset
from point_vs.preprocessing.data_loaders import SynthPharmDataset
from point_vs.dataset_generation.types_to_parquet import parse_types_mp
from point_vs.utils import load_yaml
from point_vs.utils import mkdir
import argparse
from pathlib import Path

import wandb

from point_vs.models.load_model import load_model
from point_vs.preprocessing.data_loaders import (
    get_data_loader,
    PygPointCloudDataset,
    PointCloudDataset,
)
from point_vs.utils import expand_path


def load_csv(csv_file):
    df = pd.read_csv(csv_file)
    protein_files = df["protein"]
    ligand_files = df["ligand"]
    keys = df["key"]
    pks = df["pk"]
    return protein_files, ligand_files, keys, pks


def csv_file_to_types_file(csv_file, data_dir):
    protein_files, ligand_files, keys, pks = load_csv(csv_file)
    str = ""
    for i in range(len(protein_files)):
        line = f"{pks[i]} -1 -1 {protein_files[i].split('.')[0] + '.parquet'} {ligand_files[i].split('.')[0] + '.parquet'}\n"
        str += line
    if not os.path.exists(f"data/features/{csv_file.split('/')[-1].split('.')[0]}"):
        os.makedirs(f"datafeatures/{csv_file.split('/')[-1].split('.')[0]}")
    with open(
        f"data/features/{csv_file.split('/')[-1].split('.')[0]}/data.types", "w"
    ) as f:
        f.write(str)
    return str


def convert_files_to_parquet(csv_file, data_dir):
    parse_types_mp(
        f"data/features/{csv_file.split('/')[-1].split('.')[0]}/data.types",
        Path(data_dir).expanduser(),
        f"data/features/{csv_file.split('/')[-1].split('.')[0]}/",
        True,
        mol2=False,
    )


def train_model(args):
    utils.set_gpu_mode(True)
    save_path = Path(f"temp_models/{args.model_name}").expanduser()
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "cmd_args.yaml", "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f)
    if args.model == "egnn":
        model_class = SartorrasEGNN
    elif args.model == "lucid":
        model_class = PygLucidEGNN
    else:
        raise NotImplementedError("model must be one of egnn or lucid")
    dataset_class = PygPointCloudDataset

    dl_kwargs = {
        "batch_size": args.batch_size,
        "compact": args.compact,
        "radius": args.radius,
        "use_atomic_numbers": args.use_atomic_numbers,
        "rot": False,
        "polar_hydrogens": args.hydrogens,
        "fname_suffix": args.input_suffix,
        "edge_radius": args.edge_radius,
        "estimate_bonds": args.estimate_bonds,
        "prune": args.prune,
        "extended_atom_types": args.extended_atom_types,
        "model_task": args.model_task,
        "include_strain_info": args.include_strain_info,
    }

    train_dl = get_data_loader(
        f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}/',
        dataset_class,
        augmented_actives=args.augmented_actives,
        min_aug_angle=args.min_aug_angle,
        max_active_rms_distance=args.max_active_rmsd,
        min_inactive_rms_distance=args.min_inactive_rmsd,
        max_inactive_rms_distance=args.max_inactive_rmsd,
        types_fname=f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}/data.types',
        mode="train",
        p_noise=args.p_noise,
        p_remove_entity=args.p_remove_entity,
        **dl_kwargs,
    )

    dim_input = train_dl.dataset.feature_dim

    test_dl = None
    if args.val_csv_file is not None:
        test_dl = get_data_loader(
            f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}',
            dataset_class,
            types_fname=f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}/data.types',
            mode="val",
            **dl_kwargs,
        )
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(
            train_dl.dataset,
            [
                int(0.9 * len(train_dl.dataset)),
                len(train_dl.dataset) - int(0.9 * len(train_dl.dataset)),
            ],
        )
        train_dl = GeoDataLoader(
            train_dataset,
            args.batch_size,
            False,
            sampler=None,
            drop_last=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        test_dl = GeoDataLoader(
            test_dataset,
            args.batch_size,
            False,
            sampler=None,
            drop_last=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )

    args_to_record = vars(args)

    model_kwargs = {
        "act": args.activation,
        "bn": True,
        "cache": False,
        "ds_frac": 1.0,
        "k": args.channels,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "dim_input": dim_input,
        # "dim_output": 3 if REGRESSION_TASK == "multi_regression" else 1,
        "dim_output": 1,
        "norm_coords": args.norm_coords,
        "norm_feats": args.norm_feats,
        "thin_mlps": args.thin_mlps,
        "edge_attention": args.egnn_attention,
        "attention": args.egnn_attention,
        "tanh": args.egnn_tanh,
        "normalize": args.egnn_normalise,
        "residual": args.egnn_residual,
        "edge_residual": args.egnn_edge_residual,
        "graphnorm": args.graphnorm,
        "multi_fc": args.multi_fc,
        "update_coords": not args.static_coords,
        "node_final_act": args.lucid_node_final_act,
        "permutation_invariance": args.permutation_invariance,
        "attention_activation_fn": args.attention_activation_function,
        "node_attention": args.node_attention,
        "gated_residual": args.gated_residual,
        "rezero": args.rezero,
        "model_task": args.model_task,
        "include_strain_info": args.include_strain_info,
        "final_softplus": args.final_softplus,
        "softmax_attention": args.softmax_attention,
    }

    args_to_record.update(model_kwargs)

    model = model_class(
        save_path,
        args.learning_rate,
        args.weight_decay,
        wandb_project=args.wandb_project,
        use_1cycle=args.use_1cycle,
        warm_restarts=args.warm_restarts,
        only_save_best_models=args.only_save_best_models,
        optimiser=args.optimiser,
        **model_kwargs,
    )

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    model.train_model(
        train_dl,
        epochs=args.epochs_affinity,
        top1_on_end=args.top1,
        epoch_end_validation_set=test_dl if args.val_on_epoch_end else None,
    )
    if test_dl is not None:
        model.val(test_dl, top1_on_end=args.top1)


def convert_results_file(results_fname):
    with open(
        os.path.dirname(results_fname)
        + "/affinity_"
        + str(results_fname).split("/")[-1],
        "r",
    ) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    pred = []
    true = []
    keys = []
    for l in lines:
        pred.append(float(l.split()[2]))
        true.append(float(l.split()[0]))
        keys.append(str(l.split()[4].split("/")[-1].split("_")[0]))
    return pred, true, keys


def predict(args):
    # checkpoint_path = expand_path(args.model_checkpoint)
    (
        checkpoint_path,
        model,
        model_kwargs,
        cmd_line_args,
        test_dl,
    ) = get_model_and_test_dl(
        f"data/models/{args.model_name}",
        f"data/features/{args.val_csv_file.split('/')[-1].split('.')[0]}/data.types",
        f"data/features/{args.val_csv_file.split('/')[-1].split('.')[0]}/",
    )

    results_fname = expand_path(
        Path(
            checkpoint_path.parents[1],
            "predictions_{0}-{1}.txt".format(
                Path(f"{args.val_csv_file.split('/')[-1].split('.')[0]}")
                .with_suffix("")
                .name,
                checkpoint_path.with_suffix("").name,
            ),
        )
    )

    args_to_record = vars(args)

    wandb_project = args.wandb_project
    wandb_run = args.wandb_run
    if wandb_project is not None:
        if wandb_project.lower() == "same":
            wandb_project = cmd_line_args["wandb_project"]
        if wandb_run is not None:
            if wandb_run.lower() == "same":
                wandb_run = (
                    cmd_line_args["wandb_run"]
                    + "_VAL-"
                    + Path(
                        f"data/features/{args.val_csv_file.split('/')[-1].split('.')[0]}/data.types"
                    )
                    .with_suffix("")
                    .name
                )

    args_to_record["wandb_project"] = wandb_project
    args_to_record["wandb_run"] = wandb_run

    save_path = cmd_line_args["save_path"]
    if wandb_project is not None and wandb_run is not None:
        save_path = Path(save_path, wandb_project, wandb_run)

    wandb_init_kwargs = {
        "project": wandb_project,
        "allow_val_change": True,
        "config": args_to_record,
        "dir": save_path,
    }
    if wandb_project is not None:
        wandb.init(**wandb_init_kwargs)
        if wandb_run is not None:
            wandb.run.name = wandb_run
    model = model.eval()
    model.val(test_dl, results_fname)
    pred, true, keys = convert_results_file(results_fname)
    return pred, true, keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--val_csv_file", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        help="Type of point cloud network to use: " "lucid or egnn",
        default="egnn",
    )
    parser.add_argument(
        "--train_data_root_pose",
        type=str,
        help="Location relative to which parquets files for "
        "training the pose classifier as specified in the "
        "train_types_pose file are stored.",
        default="data",
    )
    parser.add_argument(
        "--train_data_root_affinity",
        "--tdra",
        type=str,
        help="Location relative to which parquets files for "
        "training the affinity predictor as specified in "
        "the train_types file are stored.",
    )
    parser.add_argument(
        "--test_data_root_pose",
        type=str,
        help="Location relative to which parquets files for "
        "testing the pose classifier as specified in the "
        "test_types_pose file are stored.",
    )
    parser.add_argument(
        "--test_data_root_affinity",
        type=str,
        help="Location relative to which parquets files for "
        "testing the affinity predictor as specified in "
        "the test_types file are stored.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Directory in which experiment outputs are "
        "stored. If wandb_run and wandb_project are "
        "specified, save_path/wandb_project/wandb_run "
        "will be used to store results.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="info",
        help="Level at which to print logging statements. Any "
        "of notset, debug, info, warning, error, critical.",
    )
    parser.add_argument(
        "--load_weights",
        "-l",
        type=str,
        required=False,
        help="Load a model.",
        default="48L_compact__0/checkpoints/ckpt_epoch_7.pt",
    )
    parser.add_argument(
        "--translated_actives",
        type=str,
        help="Directory in which translated actives are stored."
        " If unspecified, no translated actives will be "
        "used. The use of translated actives are is "
        "discussed in https://pubs.acs.org/doi/10.1021/ac"
        "s.jcim.0c00263",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        required=False,
        default=16,
        help="Number of examples to include in each batch for " "training.",
    )
    parser.add_argument(
        "--epochs_pose",
        "-ep",
        type=int,
        required=False,
        default=0,
        help="Number of times to iterate through pose " "training set.",
    )
    parser.add_argument(
        "--epochs_affinity",
        "-ea",
        type=int,
        required=False,
        default=10,
        help="Number of times to iterate through affinity " "training set.",
    )
    parser.add_argument(
        "--channels", "-k", type=int, default=32, help="Channels for feature vectors"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.0008,
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "--weight_decay",
        "-w",
        type=float,
        default=1e-4,
        help="Weight decay for regularisation",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Name of wandb project. If left blank, wandb " "logging will not be used.",
    )
    parser.add_argument("--wandb_run", type=str, help="Name of run for wandb logging.")
    parser.add_argument(
        "--layers", type=int, default=48, help="Number of layers in LieResNet"
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=6,
        help="Maximum distance from a ligand atom for a "
        "receptor atom to be included in input",
    )
    parser.add_argument(
        "--load_args",
        type=str,
        help="Load yaml file with command line args. Any args "
        "specified in the file will overwrite other args "
        "specified on the command line.",
    )
    parser.add_argument(
        "--double", action="store_true", help="Use 64-bit floating point precision"
    )
    parser.add_argument(
        "--activation", type=str, default="relu", help="Activation function"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Chance for nodes to be inactivated on each " "trainin batch (EGNN)",
    )
    parser.add_argument(
        "--use_1cycle", action="store_true", help="Use 1cycle learning rate scheduling"
    )
    parser.add_argument(
        "--warm_restarts",
        action="store_true",
        help="Use cosine annealing with warm restarts",
        default=True,
    )
    parser.add_argument(
        "--fourier_features",
        type=int,
        default=0,
        help="(Lucid) Number of fourier terms to use when "
        "encoding distances (default is not to use "
        "fourier distance encoding)",
    )
    parser.add_argument(
        "--norm_coords",
        action="store_true",
        help="(Lucid) Normalise coordinate vectors",
        default=True,
    )
    parser.add_argument(
        "--norm_feats",
        action="store_true",
        help="(Lucid) Normalise feature vectors",
        default=True,
    )
    parser.add_argument(
        "--use_atomic_numbers",
        action="store_true",
        help="Use atomic numbers rather than smina types",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact rather than true one-hot encodings",
        default=True,
    )
    parser.add_argument(
        "--thin_mlps",
        action="store_true",
        help="(Lucid) Use single layer MLPs for edge, node and " "coord updates",
    )
    parser.add_argument(
        "--hydrogens", action="store_true", help="Include polar hydrogens"
    )
    parser.add_argument(
        "--augmented_actives",
        type=int,
        default=0,
        help="Number of randomly rotated actives to be "
        "included as decoys during training",
    )
    parser.add_argument(
        "--min_aug_angle",
        type=float,
        default=30,
        help="Minimum angle of rotation for augmented actives "
        "as specified in the augmented_actives argument",
    )
    parser.add_argument(
        "--max_active_rmsd",
        type=float,
        help="(Pose selection) maximum non-aligned RMSD "
        "between the original crystal pose and active "
        "redocked poses",
        default=2,
    )
    parser.add_argument(
        "--min_inactive_rmsd",
        type=float,
        help="(Pose selection) minimum non-aligned RMSD "
        "between original crystal pose and inactive "
        "redocked poses",
        default=2,
    )
    parser.add_argument(
        "--val_on_epoch_end",
        "-v",
        action="store_true",
        help="Run inference ion the validation set at the end "
        "of every epoch during training",
        default=True,
    )
    parser.add_argument(
        "--synth_pharm",
        "-p",
        action="store_true",
        help="Synthetic Pharmacophore mode (for Tom, beta)",
    )
    parser.add_argument(
        "--input_suffix",
        "-s",
        type=str,
        default="parquet",
        help="Filename extension for inputs",
    )
    parser.add_argument(
        "--train_types_pose",
        type=str,
        help="Optional name of GNINA-like types file which "
        "contains paths and labels for a pose training "
        "set. "
        "See GNINA 1.0 documentation for specification.",
    )
    parser.add_argument(
        "--train_types_affinity",
        type=str,
        help="Optional name of GNINA-like types file which "
        "contains paths and labels for an affinity "
        "training set. "
        "See GNINA 1.0 documentation for specification.",
    )
    parser.add_argument(
        "--test_types_pose",
        type=str,
        help="Optional name of GNINA-like types file which "
        "contains paths and labels for a pose test set. "
        "See GNINA 1.0 documentation for specification.",
    )
    parser.add_argument(
        "--test_types_affinity",
        type=str,
        help="Optional name of GNINA-like types file which "
        "contains paths and labels for an affinity test set. "
        "See GNINA 1.0 documentation for specification.",
    )
    parser.add_argument(
        "--egnn_attention",
        action="store_true",
        help="Use attention mechanism on edges for EGNN",
        default=True,
    )
    parser.add_argument(
        "--egnn_tanh",
        action="store_true",
        help="Put tanh layer at the end of the coordinates " "mlp (EGNN)",
        default=True,
    )
    parser.add_argument(
        "--egnn_normalise",
        action="store_true",
        help="Normalise radial coordinates (EGNN)",
        default=True,
    )
    parser.add_argument(
        "--egnn_residual",
        action="store_true",
        help="Use residual connections (EGNN)",
        default=True,
    )
    parser.add_argument(
        "--edge_radius",
        type=float,
        default=10,
        help="Maximum interatomic distance for an edge to " "exist (EGNN)",
    )
    parser.add_argument(
        "--end_flag",
        action="store_true",
        help='Add a file named "_FINISHED" to the save_path '
        "upon training and test completion",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        help="Location to store wandb files. Defaults to "
        "<save_path>/<wandb_project>/<wandb_run>/wandb.",
    )
    parser.add_argument(
        "--estimate_bonds",
        action="store_true",
        help="(EGNN): Instead of using a fixed edge radius,"
        "the intermolecular radius is set at "
        "--edge_radius Angstroms but the intramolecular "
        "radius is set at 2A, which has the effect of "
        "putting edges where there are covalent bonds "
        "between atoms in the same molecule.",
        default=True,
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="(EGNN) Prune subgraphs which are not connected " "to the ligand",
    )
    parser.add_argument("--top1", action="store_true", help="A poorly kept secret ;)")
    parser.add_argument(
        "--graphnorm",
        action="store_true",
        help="(EGNN) add GraphNorm layers to each node MLP",
    )
    parser.add_argument(
        "--multi_fc",
        action="store_true",
        help="Three fully connected layers rather than just "
        "one to summarise the graph at the end of "
        "the EGNN",
    )
    parser.add_argument(
        "--lucid_node_final_act",
        action="store_true",
        help="(Lucid) SiLU at the end of node MLPs",
    )
    parser.add_argument(
        "--p_remove_entity",
        type=float,
        default=0,
        help="Rate at which one of (randomly selected) ligand "
        "or receptor is removed and label is forced to "
        "zero",
    )
    parser.add_argument(
        "--static_coords",
        action="store_true",
        help="Do not update coords (eq. 4, EGNN)",
    )
    parser.add_argument(
        "--permutation_invariance",
        action="store_true",
        help="Edge features are invariant to order of input "
        "node (EGNN, experimental)",
    )
    parser.add_argument(
        "--node_attention",
        action="store_true",
        help="Use attention mechanism for nodes",
    )
    parser.add_argument(
        "--attention_activation_function",
        type=str,
        default="sigmoid",
        help="One of sigmoid, relu, silu " "or tanh",
    )
    parser.add_argument(
        "--only_save_best_models",
        action="store_true",
        help="Only save models which improve upon previous " "models",
        default=True,
    )
    parser.add_argument(
        "--egnn_edge_residual",
        action="store_true",
        help="Residual connections for individual messages " "(EGNN)",
    )
    parser.add_argument(
        "--gated_residual",
        action="store_true",
        help="Residual connections are gated by a single "
        "learnable parameter (EGNN), see "
        "home.ttic.edu/~savarese/savarese_files/Residual_Gates.pdf",
    )
    parser.add_argument(
        "--rezero",
        action="store_true",
        help="ReZero residual connections (EGNN), see " "arxiv.org/pdf/2003.04887.pdf",
    )
    parser.add_argument(
        "--extended_atom_types",
        action="store_true",
        help="18 atom types rather than 10",
    )
    parser.add_argument(
        "--max_inactive_rmsd",
        type=float,
        help="Discard structures beyond <x> RMSD from xtal " "pose",
        default=2,
    )
    parser.add_argument(
        "--model_task",
        type=str,
        default="regression",
        help="One of either classification or regression; ",
    )
    parser.add_argument("--synthpharm", action="store_true", help="For tom")
    parser.add_argument(
        "--p_noise",
        type=float,
        default=-1,
        help="Probability of label being inverted during " "training",
    )
    parser.add_argument(
        "--include_strain_info",
        action="store_true",
        help="Include info on strain energy and RMSD from " "ground state of ligand",
    )
    parser.add_argument(
        "--final_softplus",
        action="store_true",
        help="Final layer in regression has softplus " "nonlinearity",
    )
    parser.add_argument(
        "--optimiser",
        "-o",
        type=str,
        default="adam",
        help="Optimiser (either adam or sgd)",
    )
    parser.add_argument(
        "--multi_target_affinity",
        action="store_true",
        help="Use multitarget regression for affinity. If "
        "True, targets are split depending on if labels "
        "are pkd, pki or IC50.",
    )
    parser.add_argument(
        "--regression_loss", type=str, default="mse", help="Either mse or huber."
    )
    parser.add_argument(
        "--softmax_attention",
        action="store_true",
        help="Attention scores go through softmax to normalise "
        "rather than individual sigmoids.",
    )
    args = parser.parse_args()
    if args.train:
        if not os.path.exists(
            f"data/features/{args.csv_file.split('/')[-1].split('.')[0]}/"
        ):
            csv_file_to_types_file(args.csv_file, args.data_dir)
            convert_files_to_parquet(args.csv_file, args.data_dir)
        if args.val_csv_file is not None:
            if not os.path.exists(
                f"data/features/{args.val_csv_file.split('/')[-1].split('.')[0]}/"
            ):
                csv_file_to_types_file(args.val_csv_file, args.val_data_dir)
                convert_files_to_parquet(args.val_csv_file, args.val_data_dir)
        train_model(args)
    elif args.predict:
        if not os.path.exists(
            f"data/features/{args.val_csv_file.split('/')[-1].split('.')[0]}/"
        ):
            csv_file_to_types_file(args.val_csv_file, args.val_data_dir)
            convert_files_to_parquet(args.val_csv_file, args.val_data_dir)
        pred, true, keys = predict(args)
        df = pd.DataFrame({"key": keys, "pred": pred, "pk": true})
        df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
            index=False,
        )
