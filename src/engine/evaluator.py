import os
import csv
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from src.utils.plots import plot_confusion_matrix
from src.mlflow.logger import log_metrics


def evaluate(model,dataloader,device,cfg,logger,encoder,split_name="test"):
    """
    Generic evaluation engine.
    Used for test / val / external datasets.
    """

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    classes = encoder.classes_
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    logger.info(f"{split_name.upper()} Accuracy: {acc:.4f}")
    logger.info(f"{split_name.upper()} F1-score: {f1:.4f}")
    logger.info(f"{split_name.upper()} Classification Report:\n{report}")

    log_metrics({
        f"{split_name}_acc": acc,
        f"{split_name}_f1": f1
    })

    # Outputs
    plots_dir = os.path.join(cfg["output"]["results_dir"], "plots")
    tables_dir = os.path.join(cfg["output"]["results_dir"], "tables")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    plot_confusion_matrix(
        all_labels,
        all_preds,
        classes,
        os.path.join(
            plots_dir,
            f"confusion_matrix_{split_name}.png"
        )
    )

    with open(
        os.path.join(
            tables_dir,
            f"classification_report_{split_name}.txt"
        ),
        "w"
    ) as f:
        f.write(report)

    with open(
        os.path.join(
            tables_dir,
            f"{cfg['baseline']}_{split_name}_metrics.csv"
        ),
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow([f"{split_name}_acc", acc])
        writer.writerow([f"{split_name}_f1", f1])

    return {"acc": acc,"f1": f1}
