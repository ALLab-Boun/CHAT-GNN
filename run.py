import argparse
import json

import train
from util import Args


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--device", type=str, help="")

    return parser.parse_args()


def prepare_args(args):
    config_path = args.config_path
    with open(config_path) as f:
        config = json.load(f)
    base_args = vars(train.parse_arguments([]))
    final_args = base_args.copy()
    config = config["config"]
    config["n_runs"] = 10
    if args.device:
        config["device"] = args.device
    if args.seed:
        config["seed"] = args.seed
    config["loglevel"] = "info"
    final_args.update(config)
    final_args = Args(**final_args)
    return final_args


def run(args):
    run_args = prepare_args(args)
    train.set_seed(args.seed)
    train.main(run_args)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
