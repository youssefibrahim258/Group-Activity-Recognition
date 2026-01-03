import warnings
warnings.filterwarnings("ignore")
import argparse
import yaml
from src.pipelines.B1.train_b1 import train_b1
from src.pipelines.B1.eval_b1 import eval_b1
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/B1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["baseline"] == "B1":
        train_b1(cfg)
        eval_b1(cfg)
    else:
        raise NotImplementedError(f"{cfg['baseline']} not implemented yet")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    main()
