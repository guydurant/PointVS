import logging
import os
import socket
import warnings
from pathlib import Path

import torch
import yaml
import pandas as pd

from point_vs import utils
from point_vs import log
from point_vs.inference import get_model_and_test_dl

from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.parse_args import parse_args
from point_vs.preprocessing.data_loaders import get_data_loader
from point_vs.preprocessing.data_loaders import PygPointCloudDataset
from point_vs.preprocessing.data_loaders import SynthPharmDataset
from point_vs.dataset_generation import parse_types_mp
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


def types_file_creator_affinity(receptor_list, ligand_list):
    str = ""
    for i in range(len(receptor_list)):
        line = f"{receptor_list[i]} {ligand_list[i]}\n"
        str += line
    return str


def csv_file_to_types_file(csv_file, data_dir, mode="train"):
    protein_files, ligand_files, keys, pks = load_csv(csv_file, data_dir)
    str = ""
    for i in range(len(protein_files)):
        line = f"{pks[i]} -1 -1 {protein_files[i].split('.')[0] + '.parquet'} {ligand_files[i].split('.')[0] + '.parquet'}\n"
        str += line
    with open(
        f"temp_features/{csv_file.split('/')[-1].split('.')[0]}/{mode}.types", "w"
    ) as f:
        f.write(str)
    return str


def convert_files_to_parquet(csv_file, data_dir, mode="train"):
    parse_types_mp(
        f"temp_features/{csv_file.split('/')[-1].split('.')[0]}/{mode}.types",
        Path(data_dir).expanduser(),
        f"temp_features/{csv_file.split('/')[-1].split('.')[0]}/",
        True,
        mol2=False,
    )


def train_model(args):
    utils.set_gpu_mode(True)
    save_path = Path(args.model_name).expanduser()
    save_path.mkdir(parents=True, exist_ok=True)
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
        args.train_data_root,
        dataset_class,
        augmented_actives=args.augmented_actives,
        min_aug_angle=args.min_aug_angle,
        max_active_rms_distance=args.max_active_rmsd,
        min_inactive_rms_distance=args.min_inactive_rmsd,
        max_inactive_rms_distance=args.max_inactive_rmsd,
        types_fname=args.train_types,
        mode="train",
        p_noise=args.p_noise,
        p_remove_entity=args.p_remove_entity,
        **dl_kwargs,
    )

    dim_input = train_dl.dataset.feature_dim

    test_dl = None
    if args.test_data_root is not None:
        test_dl = get_data_loader(
            args.test_data_root,
            dataset_class,
            types_fname=args.test_types,
            mode="val",
            **dl_kwargs,
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
        "dim_output": 3 if args.model_task == "multi_regression" else 1,
        "norm_coords": args.norm_coords,
        "norm_feats": args.norm_feats,
        "thin_mlps": args.thin_mlps,
        "edge_attention": args.egnn_attention,
        "attention": args.egnn_attention,
        "tanh": args.egnn_tanh,
        "normalize": args.egnn_normalise,
        "residual": args.egnn_residual,
        "edge_residual": args.egnn_edge_residual,
        "linear_gap": args.linear_gap,
        "graphnorm": args.graphnorm,
        "multi_fc": args.multi_fc,
        "update_coords": not args.static_coords,
        "node_final_act": args.lucid_node_final_act,
        "permutation_invariance": args.permutation_invariance,
        "attention_activation_fn": args.attention_activation_function,
        "node_attention": args.node_attention,
        "node_attention_final_only": args.node_attention_final_only,
        "edge_attention_final_only": args.edge_attention_final_only,
        "node_attention_first_only": args.node_attention_first_only,
        "edge_attention_first_only": args.edge_attention_first_only,
        "gated_residual": args.gated_residual,
        "rezero": args.rezero,
        "model_task": args.model_task,
        "include_strain_info": args.include_strain_info,
        "final_softplus": args.final_softplus,
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

    if args.epochs:
        model.train_model(
            train_dl,
            epochs=args.epochs,
            top1_on_end=args.top1,
            epoch_end_validation_set=test_dl if args.val_on_epoch_end else None,
        )
    if test_dl is not None:
        model.val(test_dl, top1_on_end=args.top1)


def predict(args):
    checkpoint_path = expand_path(args.model_checkpoint)
    (
        checkpoint_path,
        model,
        model_kwargs,
        cmd_line_args,
        test_dl,
    ) = get_model_and_test_dl(checkpoint_path, args.test_types, args.test_data_root)

    results_fname = expand_path(
        Path(
            checkpoint_path.parents[1],
            "predictions_{0}-{1}.txt".format(
                Path(args.test_types).with_suffix("").name,
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
                    + Path(args.test_types).with_suffix("").name
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
