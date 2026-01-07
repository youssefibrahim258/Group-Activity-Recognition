import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from src.datasets.volleyball_clip_dataset import VolleyballB1Dataset
from src.models.b1_resnet import ResNetB1
from src.utils.label_encoder import LabelEncoder
from src.utils.set_seed import set_seed
from src.utils.logger import setup_logger
from src.utils.plots import plot_confusion_matrix, plot_train_loss, plot_val_f1
from src.utils.mixup import mixup_data, mixup_criterion


import matplotlib.pyplot as plt
import torch


def train_b1(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Directories =====
    os.makedirs(cfg["output"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "plots"), exist_ok=True)

    logger = setup_logger(os.path.join(cfg["output"]["results_dir"], "logs"), "train")
    writer = SummaryWriter(log_dir=os.path.join(cfg["output"]["results_dir"], "tensorboard"))

    logger.info(f"Starting training: {cfg['baseline']}")
    logger.info(f"Using device: {device}")


    # ===== Augmentations =====
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])



    transform_val = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    # ===== Dataset & DataLoader =====
    encoder = LabelEncoder(class_names=cfg["labels"]["class_names"])

    videos_root = os.path.join(cfg["data"]["videos_dir"], "videos")
    pickle_file = cfg["data"]["annot_file"]

    # Train Dataset
    train_dataset = VolleyballB1Dataset(
        pickle_file,
        videos_root,
        video_list=[str(v) for v in cfg["data"]["splits"]["train"]],
        encoder=encoder,
        transform=transform_train
    )

    # Validation Dataset
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
        pin_memory=True)


    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )



    # ===== Model =====
    model = ResNetB1(freeze_backbone=True).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable params: {total_trainable_params}")

    # ===== Loss, Optimizer, Scheduler =====
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
    lr=cfg["training"]["lr"],
    weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


    # ===== Training loop =====
    best_val_f1 = float("-inf")
    patience = 5
    early_stop_counter = 0

    train_loss_history, val_loss_history, val_f1_history = [], [], []

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixup for training loss
            imgs_mixed, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=1.0)
            outputs_mixed = model(imgs_mixed)
            loss = mixup_criterion(criterion, outputs_mixed, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Train F1: compute on original imgs/labels (no mixup)
            with torch.no_grad():
                outputs_real = model(imgs)
                train_preds.extend(torch.argmax(outputs_real, 1).cpu().tolist())
                train_labels.extend(labels.cpu().tolist())

        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        train_loss_history.append(train_loss)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, 1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)
        scheduler.step(val_f1)

        # ===== Logging =====
        logger.info(
            f"[Epoch {epoch+1}/{cfg['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)

        # ===== Early stopping =====
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(cfg["output"]["checkpoints_dir"], "best.pt")
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            logger.info(f"No improvement for {patience} epochs. Early stopping.")
            break

    # ===== Final evaluation =====
    model.load_state_dict(
        torch.load(os.path.join(cfg["output"]["checkpoints_dir"], "best.pt")),
        strict=False
    )
    model.eval()

    final_preds, final_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            final_preds.extend(torch.argmax(outputs, 1).cpu().tolist())
            final_labels.extend(labels.cpu().tolist())

    classes = encoder.classes_
    report = classification_report(
        final_labels,
        final_preds,
        labels=list(range(len(classes))),
        target_names=classes
    )
    logger.info("Final Validation Classification Report:\n" + report)

    # ===== Plots =====
    plot_confusion_matrix(
        final_labels,
        final_preds,
        classes,
        save_path=os.path.join(cfg["output"]["results_dir"], "plots", "confusion_matrix_val.png")
    )
    plot_train_loss(
        train_loss_history,
        os.path.join(cfg["output"]["results_dir"], "plots", "train_loss_curve.png")
    )
    plot_val_f1(
        val_f1_history,
        os.path.join(cfg["output"]["results_dir"], "plots", "val_f1_curve.png")
    )

    writer.close()
    logger.info("Training completed successfully.")

