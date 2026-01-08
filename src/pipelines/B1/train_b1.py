import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

from src.datasets.volleyball_clip_dataset import VolleyballB1Dataset
from src.models.b1_resnet import ResNetB1
from src.utils.label_encoder import LabelEncoder
from src.utils.set_seed import set_seed
from src.utils.logger import setup_logger
from src.engine.trainer import train
from src.mlflow.logger import start_mlflow, end_mlflow


def train_b1(cfg):
    
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Directories
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["checkpoints_dir"], exist_ok=True)

    logger = setup_logger(
        os.path.join(cfg["output"]["results_dir"], "logs"),
        "train_b1"
    )

    writer = SummaryWriter(
        log_dir=os.path.join(cfg["output"]["results_dir"], "tensorboard")
    )

    logger.info("Starting B1 Training")
    logger.info(f"Device: {device}")

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset & Dataloader
    encoder = LabelEncoder(cfg["labels"]["class_names"])

    videos_root = os.path.join(cfg["data"]["videos_dir"], "videos")
    pickle_file = cfg["data"]["annot_file"]

    train_dataset = VolleyballB1Dataset(
        pickle_file,
        videos_root,
        video_list=[str(v) for v in cfg["data"]["splits"]["train"]],
        encoder=encoder,
        transform=transform_train
    )

    val_dataset = VolleyballB1Dataset(
        pickle_file,
        videos_root,
        video_list=[str(v) for v in cfg["data"]["splits"]["val"]],
        encoder=encoder,
        transform=transform_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Class Weights
    labels_all = [label for _, label in train_dataset]
    counter = Counter(labels_all)

    num_classes = len(encoder.classes_)
    total_samples = sum(counter.values())

    class_weights = [
        total_samples / (num_classes * counter[i])
        if counter[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(device)

    logger.info(f"Train class distribution: {counter}")
    logger.info(f"Class weights: {class_weights}")

    # Model / Loss / Optim
    model = ResNetB1().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )


    if cfg["training"]["scheduler"]=="StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1
        )
    else :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"],
            eta_min=1e-6
        )

    
    # Train
    start_mlflow(cfg["baseline"], cfg["output"]["mlruns_dir"])

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        logger=logger,
        writer=writer,
        encoder=encoder,
        mixup=cfg["training"]["mixup"]
    )

    end_mlflow()
    writer.close()
    logger.info("B1 Training Finished")
