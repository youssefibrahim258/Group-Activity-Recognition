import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from src.datasets.volleyball_clip_dataset import VolleyballClipDataset
from src.models.b1_resnet import ResNetB1
from src.utils.label_encoder import LabelEncoder
from src.utils.set_seed import set_seed
from src.utils.logger import setup_logger
from src.utils.plots import plot_confusion_matrix, plot_train_loss, plot_val_f1

def train_b1(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg["output"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["output"]["results_dir"], "tensorboard"), exist_ok=True)

    logger = setup_logger(os.path.join(cfg["output"]["results_dir"], "logs"), "train")
    writer = SummaryWriter(log_dir=os.path.join(cfg["output"]["results_dir"], "tensorboard"))

    logger.info(f"Starting training: {cfg['baseline']}")
    logger.info(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    encoder = LabelEncoder(class_names=cfg["labels"]["class_names"])
    train_ds = VolleyballClipDataset(cfg["data"]["videos_dir"], cfg["data"]["splits"]["train"], encoder, transform_train)
    val_ds = VolleyballClipDataset(cfg["data"]["videos_dir"], cfg["data"]["splits"]["val"], encoder, transform_val)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = ResNetB1()
    model = model.to(device)

    # for name, param in model.named_parameters():
    #     if "layer3" in name or "layer4" in name or "classifier" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable params: {total_trainable_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Try to load previous checkpoint, ignore classifier mismatch
    # checkpoint_path = os.path.join(cfg["output"]["checkpoints_dir"], "best.pt")
    # if os.path.exists(checkpoint_path):
    #     state_dict = torch.load(checkpoint_path)
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         if k.startswith("model."):
    #             new_state_dict[k[6:]] = v
    #         else:
    #             new_state_dict[k] = v
    #     model.load_state_dict(new_state_dict, strict=False)
    #     logger.info(f"Loaded checkpoint from {checkpoint_path}")

    best_val_f1 = float('-inf')
    patience = 7
    early_stop_counter = 0
    train_loss_history, val_f1_history = [], []

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs,1).cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        train_loss_history.append(train_loss)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs,1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_f1_history.append(val_f1)
        scheduler.step(val_f1)

        logger.info(f"[Epoch {epoch+1}/{cfg['training']['epochs']}] Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(cfg["output"]["checkpoints_dir"], "best.pt"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            logger.info(f"No improvement for {patience} epochs. Early stopping.")
            break

    model.load_state_dict(torch.load(os.path.join(cfg["output"]["checkpoints_dir"], "best.pt")), strict=False)
    model.eval()

    final_preds, final_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs,1)
            final_preds.extend(preds.cpu().tolist())
            final_labels.extend(labels.cpu().tolist())

    classes = encoder.classes_
    report = classification_report(final_labels, final_preds, labels=list(range(len(classes))), target_names=classes)    
    logger.info("Final Validation Classification Report:\n" + report)

    plot_confusion_matrix(final_labels, final_preds, encoder.classes_, save_path=os.path.join(cfg["output"]["results_dir"],"plots", "confusion_matrix_val.png"))
    plot_train_loss(train_loss_history, os.path.join(cfg["output"]["results_dir"],"plots", "train_loss_curve.png"))
    plot_val_f1(val_f1_history, os.path.join(cfg["output"]["results_dir"],"plots", "val_f1_curve.png"))

    writer.close()
    logger.info("Training completed successfully.")
