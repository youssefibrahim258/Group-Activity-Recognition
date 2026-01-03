import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from src.datasets.volleyball_clip_dataset import VolleyballClipDataset
from src.models.b1_resnet import ResNetB1
from src.utils.label_encoder import LabelEncoder
from src.utils_data.annotations import load_video_annotations
from src.utils.plots import plot_confusion_matrix,plot_val_f1,plot_train_loss
from src.mlflow.logger import start_mlflow, log_params, log_metrics, end_mlflow
from src.utils.logger import setup_logger
from src.utils.set_seed import set_seed

def train_b1(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg["output"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "tables"), exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "plots"), exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "logs"), exist_ok=True)

    logger = setup_logger(os.path.join(cfg["output"]["results_dir"], "logs"), "train")
    writer = SummaryWriter(log_dir=os.path.join(cfg["output"]["results_dir"], "tensorboard"))

    logger.info(f"Starting training: {cfg['baseline']}")
    logger.info(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
                            ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    encoder = LabelEncoder(class_names=cfg["labels"]["class_names"])

    train_ds = VolleyballClipDataset(cfg["data"]["videos_dir"], cfg["data"]["splits"]["train"], encoder, transform_train)
    val_ds = VolleyballClipDataset(cfg["data"]["videos_dir"], cfg["data"]["splits"]["val"], encoder, transform_val)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = ResNetB1(cfg["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    start_mlflow(cfg["experiment"]["name"], cfg["output"]["mlruns_dir"])
    log_params({
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "lr": cfg["training"]["lr"],
        "optimizer": cfg["training"]["optimizer"],
        "weight_decay": cfg["training"]["weight_decay"]
    })

    best_val_f1 = float('-inf')
    best_ckpt = os.path.join(cfg["output"]["checkpoints_dir"], "best.pt")
    early_stop_counter = 0
    patience = 5

    train_loss_history, val_f1_history = [], []

    for epoch in range(cfg["training"]["epochs"]):
        # ---- Train ----
        model.train()
        train_preds, train_labels_epoch = [], []
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, 1).cpu().tolist())
            train_labels_epoch.extend(labels.cpu().tolist())

        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels_epoch, train_preds, average="macro")
        train_acc = accuracy_score(train_labels_epoch, train_preds)

        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, 1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = accuracy_score(val_labels, val_preds)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        log_metrics({
            "train_loss": train_loss,
            "train_f1": train_f1,
            "train_acc": train_acc,
            "val_f1": val_f1,
            "val_acc": val_acc
        }, step=epoch)
        logger.info(f"[Epoch {epoch+1}/{cfg['training']['epochs']}] Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            logger.info(f"No improvement for {patience} epochs. Stopping early.")
            break

        scheduler.step()
        train_loss_history.append(train_loss)
        val_f1_history.append(val_f1)

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    final_preds, final_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            final_preds.extend(preds.cpu().tolist())
            final_labels.extend(labels.cpu().tolist())

    classes = encoder.classes_
    report = classification_report(final_labels, final_preds, labels=list(range(len(classes))), target_names=classes)
    logger.info("Final Validation Classification Report:\n" + report)

    with open(os.path.join(cfg["output"]["results_dir"], "tables", "classification_report_val.txt"), "w") as f:
        f.write(report)

    plot_confusion_matrix(
        final_labels,
        final_preds,
        classes,
        title="Validation Confusion Matrix",
        save_path=os.path.join(cfg["output"]["results_dir"], "plots", "confusion_matrix_val_final.png"))

    csv_path = os.path.join(cfg["output"]["results_dir"], "tables", "B1_train_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["metric", "value"])
        writer_csv.writerow(["best_val_f1", best_val_f1])

    plot_train_loss(train_loss_history, os.path.join(cfg["output"]["results_dir"], "plots", "train_loss_curve.png"))
    plot_val_f1(val_f1_history, os.path.join(cfg["output"]["results_dir"], "plots", "val_f1_curve.png"))

    writer.close()
    end_mlflow()
    logger.info("Training completed successfully.")
