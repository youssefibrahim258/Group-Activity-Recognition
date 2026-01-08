import os
import torch
from sklearn.metrics import f1_score, classification_report

from src.utils.plots import plot_confusion_matrix, plot_loss, plot_f1
from src.mlflow.logger import log_metrics
from src.utils.mixup import mixup_data, mixup_criterion


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, cfg,
          logger, writer, encoder, mixup=False):
    """
    Generic training engine used for all baselines.
        - Training / validation loop
        - Logging
        - Early stopping
        - Saving best checkpoint
        - Supports Mixup for training loss
    """

    best_val_f1 = float("-inf")
    patience = cfg["training"].get("patience", 7)
    early_stop_counter = 0

    train_loss_history = []
    val_loss_history = []
    val_f1_history = []

    checkpoints_dir = cfg["output"]["checkpoints_dir"]
    plots_dir = os.path.join(cfg["output"]["results_dir"], "plots")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if mixup:
                # Mixup for loss
                imgs_mixed, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=1.0)
                outputs = model(imgs_mixed)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                # Forward pass on original images for F1
                with torch.no_grad():
                    outputs_real = model(imgs)
                    train_preds.extend(outputs_real.argmax(1).cpu().tolist())
                    train_labels.extend(labels.cpu().tolist())
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                train_preds.extend(outputs.argmax(1).cpu().tolist())
                train_labels.extend(labels.cpu().tolist())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        train_loss_history.append(train_loss)

        # Validation 
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
                val_preds.extend(outputs.argmax(1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)

        if scheduler is not None:
            scheduler.step()

        # Logging 
        logger.info(
            f"[Epoch {epoch+1}/{cfg['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )

        log_metrics(
            {
                "train_loss": train_loss,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1
            },
            step=epoch
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)

        # Early stopping 
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, "best.pt")
            )
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            logger.info("Early stopping triggered")
            break

    #  Final Evaluation 
    model.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, "best.pt")),
        strict=False
    )
    model.eval()

    final_preds, final_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            final_preds.extend(outputs.argmax(1).cpu().tolist())
            final_labels.extend(labels.cpu().tolist())

    classes = encoder.classes_

    report = classification_report(
        final_labels,
        final_preds,
        labels=list(range(len(classes))),
        target_names=classes
    )

    logger.info("Final Validation Classification Report:\n" + report)

    # Plots 
    plot_confusion_matrix(
        final_labels,
        final_preds,
        classes,
        save_path=os.path.join(cfg["output"]["results_dir"], "plots", "confusion_matrix_val.png")
    )
    plot_loss(
        train_loss_history,
        os.path.join(cfg["output"]["results_dir"], "plots", "train_loss_curve.png"),
        title="Train"
    )
    plot_loss(
        val_loss_history,
        os.path.join(cfg["output"]["results_dir"], "plots", "val_loss_curve.png"),
        title="Val"
    )

    plot_f1(
        val_f1_history,
        os.path.join(cfg["output"]["results_dir"], "plots", "val_f1_curve.png"),
        title="Val"
    )

    return model
