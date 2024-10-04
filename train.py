import argparse
import json
import os
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data import DATASET_TO_CLS, DATASET_TO_METRIC, get_dataset
from eval import evaluate
from logger import logger
from model import MODEL_TO_CLS
from util import get_device, num_parameters, set_seed

device = get_device()


def get_optimizer(model, lr, weight_decay, weight_decay_prop):
    if type(model) == MODEL_TO_CLS["aero"]:
        optimizer = torch.optim.Adam(
            [
                {"params": model.dense_lins.parameters(), "weight_decay": weight_decay},
                {"params": model.atts.parameters(), "weight_decay": weight_decay_prop},
                {
                    "params": model.hop_atts.parameters(),
                    "weight_decay": weight_decay_prop,
                },
                {
                    "params": model.hop_biases.parameters(),
                    "weight_decay": weight_decay_prop,
                },
            ],
            lr=lr,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    return optimizer


def get_model(model_name, args):
    return MODEL_TO_CLS[model_name](**args)


def train(
    model,
    dataset,
    optimizer,
    loss_func,
    epochs,
    eval_metric,
    early_stopping_rounds,
    early_stopping_start,
    split=0,
):
    best_train_eval = 0
    best_val_eval = 0
    best_train_loss = 1e10
    best_state = model.state_dict()
    early_stop_count = 0
    best_epoch = 0

    stats = {
        "train_loss": [],
        "train_eval": [],
        "test_loss": [],
        "test_eval": [],
        "val_loss": [],
        "val_eval": [],
    }
    graph = dataset[0]
    x = graph.x
    y = graph.y
    if dataset.num_classes == 2:
        y = y.to(torch.float32)
    edge_index = graph.edge_index
    if graph.train_mask.ndim > 1:
        train_mask = graph.train_mask[:, split]
    else:
        train_mask = graph.train_mask

    progress_bar = tqdm(range(epochs))
    for e in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(str(device).split(":")[0]):
            pred = model(x, edge_index)
            if dataset.num_classes == 2:
                pred = pred.squeeze()
            loss = loss_func(pred[train_mask], y[train_mask])
        loss.backward()

        optimizer.step()

        eval_dict = evaluate(
            model,
            dataset,
            split,
            loss_func,
            eval_metric=eval_metric,
        )

        stats["train_loss"].append(eval_dict["train_loss"])
        stats["train_eval"].append(eval_dict["train_eval"])
        stats["test_loss"].append(eval_dict["test_loss"])
        stats["test_eval"].append(eval_dict["test_eval"])
        stats["val_loss"].append(eval_dict["val_loss"])
        stats["val_eval"].append(eval_dict["val_eval"])

        train_eval = eval_dict["train_eval"]
        val_eval = eval_dict["val_eval"]

        if best_val_eval < val_eval:
            best_val_eval = val_eval
            best_state = deepcopy(model.state_dict())
            early_stop_count = 0
            best_epoch = e
        else:
            if e >= early_stopping_start:
                early_stop_count += 1
            if early_stop_count >= early_stopping_rounds:
                break

        if best_train_eval < train_eval:
            best_train_eval = train_eval

        if best_train_loss > loss:
            best_train_loss = loss.item()

        model.train()

        if (e + 1) % 50 == 0:
            progress_bar.set_postfix(
                {
                    "best_epoch": best_epoch,
                    "train_loss": loss.item(),
                    "best_loss": best_train_loss,
                    "train_eval": train_eval,
                    "best_train_eval": best_train_eval,
                    "val_eval": val_eval,
                    "best_val_eval": best_val_eval,
                }
            )
    sec_per_epoch = (progress_bar.last_print_t - progress_bar.start_t) / e
    stats["sec_per_epoch"] = round(sec_per_epoch, 2)
    model.load_state_dict(best_state)
    return model, stats


def run_experiment(args, dataset):
    train_evals = []
    val_evals = []
    test_evals = []
    if dataset._data.train_mask.ndim > 1:
        num_splits = dataset._data.train_mask.shape[-1]
    else:
        num_splits = 1

    model_args = {
        "in_features": dataset.num_features,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "out_features": 1 if dataset.num_classes == 2 else dataset.num_classes,
        "dropout": args.dropout,
        "initial_dropout": args.initial_dropout,
        "n_heads": args.n_heads,
        "multi_out_heads": args.multi_out_heads,
        "activation_func": args.activation,
        "aggr": args.aggr,
        "jk_mode": args.jk_mode,
        "iterations": args.iterations,
        "lambd": args.lambd,
        "add_dropout": args.add_dropout,
        "alpha": args.alpha,
        "rgnn_prop": args.rgnn_prop,
        "rgnn_rnn": args.rgnn_rnn,
        "eps": args.eps,
        "alpha_gcn2": args.alpha_gcn2,
        "theta_gcn2": args.theta_gcn2,
        "return_attention_weights": args.return_attention_weights,
        "cn_scale": args.cn_scale,
        "cn_tau": args.cn_tau,
        "dir_alpha": args.dir_alpha,
        "dir_conv": args.dir_conv,
        "og_chunk_size": args.og_chunk_size,
        "og_num_input_layers": args.og_num_input_layers,
        "og_simple_gating": args.og_simple_gating,
        "og_global_gating": args.og_global_gating,
        "post_transform": args.post_transform,
        "layer_norm": args.layer_norm,
        "post_transform_shared": args.post_transform_shared,
        "ca_activation": args.ca_activation,
    }
    model_name = args.model

    for i in range(args.n_runs):
        split = i % num_splits

        model = get_model(model_name, model_args)
        model = model.to(device)
        if i == 0:
            logger.info(model)
            logger.info(f"Num. parameters: {num_parameters(model)}")
        optimizer = get_optimizer(
            model, args.lr, args.weight_decay, args.weight_decay_prop
        )
        if model_args["out_features"] == 1:
            loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        model, stats = train(
            model,
            dataset,
            optimizer,
            loss_func,
            epochs=args.epochs,
            eval_metric=DATASET_TO_METRIC[args.dataset],
            early_stopping_rounds=args.early_stopping_rounds,
            early_stopping_start=args.early_stopping_start,
            split=split,
        )

        eval_dict = evaluate(
            model, dataset, split, eval_metric=DATASET_TO_METRIC[args.dataset]
        )
        train_eval = eval_dict["train_eval"]
        test_eval = eval_dict["test_eval"]
        val_eval = eval_dict["val_eval"]
        train_evals.append(train_eval)
        val_evals.append(val_eval)
        test_evals.append(test_eval)
        logger.info(
            f"""Best model evaluation:
                    (Run {i+1}/{args.n_runs})
                    Train: {np.round(train_eval * 100, 1)}
                    Validation: {np.round(val_eval * 100, 1)}
                    Test: {np.round(test_eval* 100, 1)}"""
        )

    train_mean = np.round(np.mean(train_evals) * 100, 1)
    train_std = np.round(np.std(train_evals) * 100, 1)
    val_mean = np.round(np.mean(val_evals) * 100, 1)
    val_std = np.round(np.std(val_evals) * 100, 1)
    test_mean = np.round(np.mean(test_evals) * 100, 1)
    test_std = np.round(np.std(test_evals) * 100, 1)
    logger.info(
        """Test mean: {} Test std: {}""".format(
            test_mean,
            test_std,
        )
    )

    if args.output_name:
        eval_dict = evaluate(
            model,
            dataset,
            split=0,
            loss_func=loss_func,
            calc_mad_gap=args.calc_mad_gap,
            calc_dirichlet_energy=args.calc_dirichlet,
            eval_metric=DATASET_TO_METRIC[args.dataset],
        )

        stats["eval"] = {
            "model": args.model,
            "dataset": args.dataset,
            "model_args": model_args,
            "train_args": vars(args),
            "split": split,
            "train_mean": train_mean,
            "train_std": train_std,
            "test_mean": test_mean,
            "test_std": test_std,
            "val_mean": val_mean,
            "val_std": val_std,
            "sec_per_epoch": stats["sec_per_epoch"],
            **eval_dict,
        }
    else:
        stats = {}

    return model, stats


def report_train_stats(stats, path):
    epochs = len(stats["train_loss"])
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(epochs), stats["train_loss"], label="Train")
    plt.plot(np.arange(epochs), stats["test_loss"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(path, "loss_per_epoch.png"))

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(epochs), stats["train_eval"], label="Train")
    plt.plot(np.arange(epochs), stats["test_eval"], label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Eval")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(path, "eval_per_epoch.png"))

    json.dump(
        stats["eval"],
        open(os.path.join(path, "eval.json"), "w"),
        indent=4,
        separators=(",", ": "),
    )


def save_node_rep(model, dataset, file_path):
    graph = dataset[0]
    x = graph.x
    edge_index = graph.edge_index
    model.eval()

    reps = {}

    def store_reps(self, input, output, layer_id):
        reps[layer_id] = output.cpu().numpy()

    for i, l in enumerate(model.conv_layers):
        l.register_forward_hook(partial(store_reps, layer_id=f"layer_{i}"))

    with torch.no_grad():
        model(x, edge_index)

    torch.save(reps, file_path)


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def load_model(model_name, model_args, file_path):
    model = get_model(model_name, model_args)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cora", choices=DATASET_TO_CLS.keys(), help=""
    )
    parser.add_argument(
        "--model", type=str, default="gcn", choices=MODEL_TO_CLS.keys(), help=""
    )
    parser.add_argument("--save-model", action="store_true", help="")
    parser.add_argument("--save-node-rep", action="store_true", help="")
    parser.add_argument("--hidden-size", type=int, default=16, help="")
    parser.add_argument("--lr", type=float, default=0.01, help="")
    parser.add_argument("--dropout", type=float, default=0.5, help="")
    parser.add_argument("--initial-dropout", action="store_true", help="")
    parser.add_argument("--epochs", type=int, default=5000, help="")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="")
    parser.add_argument("--num-layers", type=int, default=2, help="")
    parser.add_argument("--n-runs", type=int, default=1, help="")
    parser.add_argument("--n-heads", type=int, default=8, help="")
    parser.add_argument("--multi-out-heads", action="store_true", help="")
    parser.add_argument("--return-attention-weights", action="store_true", help="")
    parser.add_argument("--activation", type=str, default="relu", help="")
    parser.add_argument("--aggr", type=str, default="mean", help="")
    parser.add_argument("--calc-mad-gap", action="store_true", help="")
    parser.add_argument("--calc-dirichlet", action="store_true", help="")
    parser.add_argument(
        "--jk-mode", type=str, default="max", choices=["cat", "max", "lstm"], help=""
    )
    parser.add_argument("--output-name", type=str, help="")
    parser.add_argument(
        "--loglevel", type=str, default="info", choices=["info", "error"], help=""
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=500, help="")
    parser.add_argument("--early-stopping-start", type=int, default=200, help="")
    parser.add_argument("--device", type=str)

    # AERO-GNN specific params
    parser.add_argument("--iterations", type=int, default=32, help="")
    parser.add_argument("--lambd", type=float, default=0.5, help="")
    parser.add_argument("--add-dropout", action="store_true", help="")
    parser.add_argument("--weight-decay-prop", type=float, default=0.01, help="")

    # Residual specific params
    parser.add_argument("--alpha", type=float, default=0.4, help="")  # GATv2Res
    parser.add_argument("--eps", type=float, default=0.1, help="")  # FAGCN

    # rgnn specific params
    parser.add_argument(
        "--rgnn-prop", type=str, default="gcn", choices=["gat", "gcn"], help=""
    )
    parser.add_argument(
        "--rgnn-rnn", type=str, default="gru", choices=["lstm", "gru"], help=""
    )

    # gcnii
    parser.add_argument("--alpha-gcn2", type=float, default=0.1, help="")
    parser.add_argument("--theta-gcn2", type=float, default=0.5, help="")

    # contranorm
    parser.add_argument("--cn-scale", type=float, default=0.2, help="")
    parser.add_argument("--cn-tau", type=float, default=1, help="")

    # dir gnn
    parser.add_argument("--dir-alpha", type=float, default=0.5, help="")
    parser.add_argument(
        "--dir-conv", type=str, default="gcn", choices=["gcn", "chatgnn"], help=""
    )

    # ordered gnn
    parser.add_argument("--og-chunk-size", type=int, default=64, help="")
    parser.add_argument("--og-num-input-layers", type=int, default=2, help="")
    parser.add_argument("--og-simple-gating", action="store_true", help="")
    parser.add_argument("--og-global-gating", action="store_true", help="")

    # chatgnn params
    parser.add_argument("--post-transform", action="store_true", help="")
    parser.add_argument("--layer-norm", action="store_true", help="")
    parser.add_argument("--post-transform-shared", action="store_true", help="")
    parser.add_argument(
        "--ca-activation",
        type=str,
        default="tanh",
        choices=["tanh", "sigmoid"],
        help="",
    )

    # optional seed
    parser.add_argument("--seed", type=int, help="")

    return parser.parse_args(args)


def main(args):
    if args.device:
        global device
        device = torch.device(args.device)

    logger.setLevel(args.loglevel.upper())

    dataset = get_dataset(
        args.dataset,
        device=device,
        undirected=False if args.model == "dirgnn" else True,
        one_hot=True
        if args.model == "gonn" and args.dataset in ["chameleon", "squirrel"]
        else False,
    )

    logger.info(
        f"""
        Dataset: {args.dataset}
        Nodes: {dataset._data.num_nodes}
        Edges: {dataset._data.num_edges}
        Features: {dataset.num_features}
        Classes: {dataset.num_classes}
        """
    )

    model, stats = run_experiment(args, dataset)

    if args.output_name:
        os.makedirs(os.path.join("output", args.output_name), exist_ok=True)
        if args.save_node_rep:
            save_node_rep(
                model, dataset, os.path.join("output", args.output_name, "node_rep.pt")
            )
        report_train_stats(stats, os.path.join("output", args.output_name))
        if args.save_model:
            save_model(model, os.path.join("output", args.output_name, "model.pt"))

        return stats["eval"]


if __name__ == "__main__":
    args = parse_arguments()
    if args.seed:
        set_seed(args.seed)
    main(args)
