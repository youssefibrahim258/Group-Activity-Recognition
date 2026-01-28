import warnings
warnings.filterwarnings("ignore")
import argparse
import yaml
from src.pipelines.B1.train_b1 import train_b1
from src.pipelines.B1.eval_b1 import eval_b1
from src.pipelines.B3.train_b3 import train_b3
from src.pipelines.B3.eval_b3 import eval_b3
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
    
    elif cfg["baseline"] == "B3":
        train_b3(cfg)
        eval_b3(cfg)
    else:
        raise NotImplementedError(f"{cfg['baseline']} not implemented yet")


if __name__ == "__main__":

    main()
