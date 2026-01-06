import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.datasets.volleyball_clip_dataset import VolleyballClip9FramesDataset
from src.models.b1_resnet import ResNetB1
from src.utils.label_encoder import LabelEncoder
from src.utils.plots import plot_confusion_matrix
from src.mlflow.logger import start_mlflow, log_metrics, end_mlflow
from src.utils.logger import setup_logger
from src.utils.set_seed import set_seed


def eval_b1(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.join(cfg["output"]["results_dir"], "tables"), exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "plots"), exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "logs"), exist_ok=True)

    logger = setup_logger(os.path.join(cfg["output"]["results_dir"], "logs"), "test")
    logger.info("Starting TEST evaluation")
    logger.info(f"Using device: {device}")

    # ===== Transform =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # ===== Dataset & Loader =====
    encoder = LabelEncoder(cfg["labels"]["class_names"])

    test_ds = VolleyballClip9FramesDataset(
        cfg["data"]["videos_dir"],
        cfg["data"]["splits"]["test"],
        encoder,
        transform,
        repeat=1
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"]
    )

    # ===== Model =====
    model = ResNetB1().to(device)
    checkpoint = os.path.join(cfg["output"]["checkpoints_dir"], "best.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ===== MLflow =====
    start_mlflow(cfg["baseline"] + "_test", cfg["output"]["mlruns_dir"])

    all_preds, all_labels_test = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)       # (B, C, H, W)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels_test.extend(labels.cpu().tolist())

    # ===== Metrics =====
    acc = accuracy_score(all_labels_test, all_preds)
    f1 = f1_score(all_labels_test, all_preds, average="macro")

    classes = encoder.classes_
    report = classification_report(
        all_labels_test,
        all_preds,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    logger.info(f"TEST Accuracy: {acc:.4f}")
    logger.info(f"TEST F1-score: {f1:.4f}")
    logger.info(f"Classification Report:\n{report}")

    log_metrics({
        "test_acc": acc,
        "test_f1": f1
    })

    # ===== Plots & Tables =====
    plot_confusion_matrix(
        all_labels_test,
        all_preds,
        classes,
        os.path.join(cfg["output"]["results_dir"], "plots", "confusion_matrix_test.png")
    )

    with open(
        os.path.join(cfg["output"]["results_dir"], "tables", "classification_report_test.txt"),
        "w"
    ) as f:
        f.write(report)

    with open(
        os.path.join(cfg["output"]["results_dir"], "tables", "B1_test_metrics.csv"),
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["test_acc", acc])
        writer.writerow(["test_f1", f1])

    end_mlflow()
    logger.info("TEST evaluation completed successfully")
