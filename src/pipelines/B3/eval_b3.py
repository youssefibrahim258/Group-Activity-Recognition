import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.volleyball_player_dataset import VolleyballB3Dataset
from src.models.b3_a_resnet import ResNetB3
from src.utils.label_encoder import LabelEncoder
from src.utils.logger import setup_logger
from src.utils.set_seed import set_seed
from src.engine.evaluator import evaluate
from src.mlflow.logger import start_mlflow, end_mlflow
from src.utils_data.collate_fn import my_collate_fn



def eval_b3(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger(
        os.path.join(cfg["output"]["results_dir"], "logs"),
        "eval_b3"
    )

    logger.info("Starting B3 TEST evaluation")
    logger.info(f"Device: {device}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    encoder = LabelEncoder(cfg["labels"]["class_names"])

    # Dataset & Loader
    videos_root = os.path.join(cfg["data"]["videos_dir"], "videos")
    pickle_file = cfg["data"]["annot_file"]

    test_dataset = VolleyballB3Dataset(
        pickle_file,
        videos_root,
        video_list=[str(v) for v in cfg["data"]["splits"]["test"]],
        encoder=encoder,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=my_collate_fn
    )

    # Model
    model = ResNetB3(num_classes=9).to(device)

    checkpoint = os.path.join(
        cfg["output"]["checkpoints_dir"],
        "best.pt"
    )

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(
        torch.load(checkpoint, map_location=device),
        strict=False
    )

    # MLflow
    start_mlflow(cfg["baseline"] + "_test",cfg["output"]["mlruns_dir"])

    evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        cfg=cfg,
        logger=logger,
        encoder=encoder,
        split_name="test"
    )

    end_mlflow()
    logger.info("B3 TEST evaluation finished successfully")
