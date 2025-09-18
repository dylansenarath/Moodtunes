import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset import GoEmotionsDataset
from src.model import CustomDistilBERT


# -----------------------
# Collate (tokenization)
# -----------------------
def make_collate_fn(tokenizer, max_length: int):
    def _collate(batch):
        # batch = list[(text:str, labels:tensor)]
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        labels = torch.stack(labels).to(torch.float32)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
    return _collate


# -----------------------
# Training / Eval utils
# -----------------------
def compute_pos_weight(data_loader):
    total_counts = None
    total_samples = 0
    for batch in tqdm(data_loader, desc="Computing class weights"):
        labels = batch["labels"]
        total_counts = labels.sum(dim=0) if total_counts is None else total_counts + labels.sum(dim=0)
        total_samples += labels.size(0)
    pos_weight = (total_samples - total_counts) / (total_counts + 1e-6)
    return pos_weight


def train_model(model, train_loader, val_loader, device, epochs=5,
                early_stopping_patience=1, focal_loss_gamma=1.0, label_names=None):
    model.to(device)
    sigmoid = torch.nn.Sigmoid()

    # class weights
    pos_weight = compute_pos_weight(train_loader).to(device)
    # optionally emphasize certain labels
    label_weights = torch.ones(len(label_names)).to(device)
    for i, lbl in enumerate(label_names):
        if lbl in ["joy", "neutral"]:
            label_weights[i] = 3.0
    pos_weight = pos_weight * label_weights

    use_focal = focal_loss_gamma > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    total_steps = len(train_loader) * max(epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0.0
    no_improve = 0
    best_state, best_thresholds = None, None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            if use_focal:
                probs = sigmoid(logits)
                pt = torch.where(labels == 1, probs, 1 - probs)
                focal_weight = (1 - pt) ** focal_loss_gamma
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none", pos_weight=pos_weight
                )
                loss = (focal_weight * bce).mean()
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        print(f"Training Loss: {running_loss / max(len(train_loader), 1):.4f}")

        # ---- validation ----
        model.eval()
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                probs = sigmoid(logits)
                all_outputs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # thresholds per class by PR-curve F1
        thresholds = []
        for i in range(all_outputs.shape[1]):
            precision, recall, thresh = precision_recall_curve(all_labels[:, i], all_outputs[:, i])
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            idx = int(np.argmax(f1_scores))
            thr = thresh[idx] if idx < len(thresh) else 0.5
            # adjust a few weaker labels
            if label_names[i] in ["joy", "neutral", "fear", "surprise"]:
                thr = max(min(thr, 0.5), 0.3)
            thresholds.append(thr)

        # apply thresholds
        preds = (all_outputs > np.array(thresholds)[None, :]).astype(int)

        micro_f1 = f1_score(all_labels, preds, average="micro")
        macro_f1 = f1_score(all_labels, preds, average="macro")
        weighted_f1 = f1_score(all_labels, preds, average="weighted")
        print(f"Validation - Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

        if micro_f1 > best_f1:
            best_f1 = micro_f1
            no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_thresholds = thresholds
            torch.save({
                "model_state_dict": best_state,
                "thresholds": best_thresholds,
                "config": {"num_labels": len(thresholds), "model_name": "distilbert-base-uncased"},
            }, "best_emotion_model.pt")
            print(f"New best model saved with F1: {best_f1:.4f}")

            report = classification_report(all_labels, preds, target_names=label_names, zero_division=0)
            print(report)
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s).")

        if no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epoch(s).")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_thresholds


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data", help="Folder containing dataset files")
    parser.add_argument("--labels", type=str, default="emotions.txt", help="Label list file in data-root")
    parser.add_argument("--ekman-map", type=str, default="ekman_mapping.json",
                        help="Optional Ekman mapping JSON in data-root")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    data_root = args.data_root.rstrip("/\\")
    labels_path = os.path.join(data_root, args.labels)
    ekman_path = os.path.join(data_root, args.ekman_map)

    # labels (optionally remapped to Ekman)
    label_names = load_labels(labels_path)
    mapped = None
    if os.path.exists(ekman_path):
        try:
            with open(ekman_path, "r", encoding="utf-8") as f:
                ekman_mapping = json.load(f)
            ekman_mapping["neutral"] = list(set(ekman_mapping.get("neutral", []) + ["neutral"]))
            fine_to_ekman = {fine: ek for ek, fines in ekman_mapping.items() for fine in fines}
            mapped = sorted(set(fine_to_ekman.get(lbl, lbl) for lbl in label_names))
            label_names = mapped
        except Exception as e:
            print(f"Warning: failed to apply Ekman mapping ({e}); using original labels.")

    print("Label count:", len(label_names))
    print("Labels:", label_names)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = GoEmotionsDataset(
        file_path=os.path.join(data_root, "train.tsv"),
        label_list=label_names,
        augment=True,
        balance=True,
    )
    dev_dataset = GoEmotionsDataset(
        file_path=os.path.join(data_root, "dev.tsv"),
        label_list=label_names,
        augment=False,
        balance=False,
    )

    collate = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CustomDistilBERT(model_name="distilbert-base-uncased",
                             num_labels=len(label_names),
                             dropout=0.3)

    model, thresholds = train_model(
        model, train_loader, dev_loader, device,
        epochs=args.epochs,
        early_stopping_patience=1,
        focal_loss_gamma=1.0,
        label_names=label_names,
    )

    # ---- final eval ----
    model.eval()
    sigmoid = torch.nn.Sigmoid()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Final Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            probs = sigmoid(logits)
            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    thr = np.array(thresholds) if thresholds is not None else 0.5
    if isinstance(thr, float):
        preds = (all_outputs > thr).astype(int)
    else:
        preds = (all_outputs > thr[None, :]).astype(int)

    micro = f1_score(all_labels, preds, average="micro")
    macro = f1_score(all_labels, preds, average="macro")
    print("\n=== Final Evaluation with Optimized Thresholds ===")
    print(f"Micro F1: {micro:.4f}")
    print(f"Macro F1: {macro:.4f}")
    print(classification_report(all_labels, preds, target_names=label_names, zero_division=0))

    print("\nOptimal thresholds per class:")
    if isinstance(thr, float):
        print(f"(uniform) {thr:.3f}")
    else:
        for lbl, t in zip(label_names, thr):
            print(f"{lbl}: {float(t):.3f}")


if __name__ == "__main__":
    main()
