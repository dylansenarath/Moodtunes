import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from src.model import CustomDistilBERT

# NLTK (safe import/download)
import nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    try:
        nltk.download("wordnet", quiet=True)
    except Exception:
        pass
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass
from nltk.corpus import wordnet


# -----------------------
# Dataset
# -----------------------
class EnhancedGoEmotionsDataset(Dataset):
    def __init__(self, tsv_path, emotion_labels, tokenizer=None, augment=False,
                 augment_ratio=0.3, balance_classes=False, max_length=128):
        self.data = pd.read_csv(tsv_path, sep="\t", header=None, names=["text", "label_index", "id"])

        # process labels (list of ints)
        self.label_indices = []
        for labels in self.data["label_index"]:
            if isinstance(labels, str) and "," in labels:
                self.label_indices.append([int(label) for label in labels.split(",")])
            else:
                self.label_indices.append([int(labels)])

        self.texts = [self.clean_text(str(t)) for t in self.data["text"].astype(str).tolist()]
        self.original_indices = list(range(len(self.texts)))

        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.emotion_labels = emotion_labels
        self.max_length = max_length

        if augment:
            self.augment_data(augment_ratio)
        if balance_classes:
            self.balance_classes()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
        text = re.sub(r"@\w+", " [USER] ", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = re.sub(r"([!?.]){2,}", r"\1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def augment_data(self, augment_ratio=0.3):
        num_to_augment = int(len(self.texts) * augment_ratio)

        all_labels = [label for sublist in self.label_indices for label in sublist]
        label_counts = Counter(all_labels)
        total = sum(label_counts.values())
        label_weights = {label: 1.0 - (count / max(total, 1)) for label, count in label_counts.items()}

        sample_weights = []
        for idx in range(len(self.texts)):
            weight = sum(label_weights[label] for label in self.label_indices[idx])
            sample_weights.append(weight)

        total_weight = sum(sample_weights) or 1.0
        sample_weights = [w / total_weight for w in sample_weights]

        indices_to_augment = np.random.choice(
            len(self.texts), size=num_to_augment, replace=False, p=sample_weights
        )

        augmentation_techniques = [
            self._synonym_replacement,
            self._random_swap,
            self._random_deletion,
        ]

        augmented_texts, augmented_labels = [], []
        for idx in indices_to_augment:
            augment_func = random.choice(augmentation_techniques)
            new_text = augment_func(self.texts[idx])
            augmented_texts.append(new_text)
            augmented_labels.append(self.label_indices[idx].copy())

        self.texts.extend(augmented_texts)
        self.label_indices.extend(augmented_labels)
        self.original_indices.extend([-1] * len(augmented_texts))
        print(f"Added {len(augmented_texts)} augmented samples. New dataset size: {len(self.texts)}")

    def balance_classes(self, strategy="oversample"):
        class_counts = {}
        sample_class_map = {}
        for idx, labels in enumerate(self.label_indices):
            for label in labels:
                class_counts.setdefault(label, 0)
                sample_class_map.setdefault(label, [])
                class_counts[label] += 1
                sample_class_map[label].append(idx)

        if strategy == "oversample":
            max_count = max(class_counts.values())
            majority_classes = [l for l, c in class_counts.items() if c == max_count]
            print("Majority class(es):", majority_classes, "→ count:", max_count)

            target_count = int(max_count * 0.5)  # 50% of majority

            new_texts, new_labels = [], []
            for label, count in class_counts.items():
                if count < target_count:
                    samples_needed = target_count - count
                    sampled_indices = np.random.choice(sample_class_map[label], size=samples_needed, replace=True)
                    for idx in sampled_indices:
                        new_texts.append(self.texts[idx])
                        new_labels.append(self.label_indices[idx].copy())

            self.texts.extend(new_texts)
            self.label_indices.extend(new_labels)
            self.original_indices.extend([-1] * len(new_texts))
            print(f"Added {len(new_texts)} oversampled examples. New dataset size: {len(self.texts)}")

    def _get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                s = lemma.name().replace("_", " ")
                if s != word:
                    synonyms.add(s)
        return list(synonyms)

    def _synonym_replacement(self, text, n=1):
        words = text.split()
        if len(words) <= 1:
            return text
        n = min(n, len(words))
        replace_indices = random.sample(range(len(words)), n)
        for idx in replace_indices:
            word = words[idx]
            syns = self._get_synonyms(word)
            if syns:
                words[idx] = random.choice(syns)
        return " ".join(words)

    def _random_swap(self, text, n=1):
        words = text.split()
        if len(words) <= 1:
            return text
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return " ".join(words)

    def _random_deletion(self, text, p=0.1):
        words = text.split()
        if len(words) <= 1:
            return text
        kept = [w for w in words if random.random() > p]
        if not kept:
            return random.choice(words)
        return " ".join(kept)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.label_indices[idx]
        labels = torch.zeros(len(self.emotion_labels), dtype=torch.float)
        for li in label_indices:
            if 0 <= li < len(self.emotion_labels):
                labels[li] = 1.0

        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_length
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


# -----------------------
# Model
# -----------------------
class AttentionPoolingBertClassifier(nn.Module):
    def __init__(self, num_labels, pretrained_model_name="distilbert-base-uncased", dropout_rate=0.2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)

        # (light) freeze embeddings
        for p in self.distilbert.embeddings.parameters():
            p.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.pre_classifier = nn.Linear(self.distilbert.config.hidden_size, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(512, num_labels)

        self.thresholds = torch.ones(num_labels) * 0.5

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, H)
        attn = self.attention(hidden)       # (B, T, 1)
        ctx = torch.sum(attn * hidden, dim=1)
        x = self.dropout(ctx)
        x = self.pre_classifier(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def set_thresholds(self, thresholds):
        self.thresholds = torch.tensor(thresholds)

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        if self.thresholds.device != probs.device:
            self.thresholds = self.thresholds.to(probs.device)
        preds = (probs > self.thresholds).float()
        return preds, probs


# -----------------------
# Training / Eval
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

    pos_weight = compute_pos_weight(train_loader).to(device)
    label_weights = torch.ones(len(label_names)).to(device)
    for i, label in enumerate(label_names):
        if label in ["joy", "neutral"]:
            label_weights[i] = 3.0
    pos_weight = pos_weight * label_weights

    use_focal = focal_loss_gamma > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0.0
    no_improve = 0
    best_state, best_thresholds = None, None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0

        for _, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            if use_focal:
                probs = sigmoid(outputs)
                pt = torch.where(labels == 1, probs, 1 - probs)
                focal_weight = (1 - pt) ** focal_loss_gamma
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs, labels, reduction="none", pos_weight=pos_weight
                )
                loss = (focal_weight * bce).mean()
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs, labels, pos_weight=pos_weight
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        print(f"Training Loss: {train_loss / max(len(train_loader),1):.4f}")

        # validation
        model.eval()
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                probs = sigmoid(outputs)
                all_outputs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        thresholds = []
        for i in range(all_outputs.shape[1]):
            precision, recall, thresh = precision_recall_curve(all_labels[:, i], all_outputs[:, i])
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            if optimal_idx < len(thresh):
                thresholds.append(thresh[optimal_idx])
            else:
                thresholds.append(0.5)

        preds = np.zeros_like(all_outputs, dtype=int)
        for i in range(all_outputs.shape[1]):
            label = label_names[i]
            adjusted = thresholds[i]
            if label in ["joy", "neutral", "fear", "surprise"]:
                adjusted = max(min(adjusted, 0.5), 0.3)
            preds[:, i] = (all_outputs[:, i] > adjusted).astype(int)

        micro_f1 = f1_score(all_labels, preds, average="micro")
        macro_f1 = f1_score(all_labels, preds, average="macro")
        weighted_f1 = f1_score(all_labels, preds, average="weighted")
        print(f"Validation - Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

        if micro_f1 > best_f1:
            best_f1 = micro_f1
            no_improve = 0
            best_state = model.state_dict().copy()
            best_thresholds = thresholds
            torch.save({
                "model_state_dict": best_state,
                "thresholds": best_thresholds,
                "config": {"num_labels": len(thresholds), "model_name": "distilbert-base-uncased"},
            }, "best_emotion_model.pt")
            print(f"New best model saved with F1: {best_f1:.4f}")

            if label_names:
                report = classification_report(all_labels, preds, target_names=label_names, zero_division=0)
                print(report)
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs.")

        if no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.set_thresholds(best_thresholds)
    return model, best_thresholds


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Folder containing goemotions_data/")
    parser.add_argument("--labels", type=str, default="goemotions_data/emotions.txt")
    parser.add_argument("--ekman-map", type=str, default="goemotions_data/ekman_mapping.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    data_root = args.data_root.rstrip("/\\")
    labels_path = os.path.join(data_root, args.labels)
    ekman_path = os.path.join(data_root, args.ekman_map)

    label_names = load_labels(labels_path)

    with open(ekman_path, "r", encoding="utf-8") as f:
        ekman_mapping = json.load(f)
    ekman_mapping["neutral"] = ["neutral"]
    fine_to_ekman = {}
    for ekman_cat, fines in ekman_mapping.items():
        for lbl in fines:
            fine_to_ekman[lbl] = ekman_cat
    mapped_label_names = sorted(set(fine_to_ekman.get(lbl, lbl) for lbl in label_names))
    print("Original labels:", len(label_names))
    print("Mapped Ekman labels:", mapped_label_names)
    label_names = mapped_label_names

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = EnhancedGoEmotionsDataset(
        tsv_path=os.path.join(data_root, "goemotions_data", "train.tsv"),
        emotion_labels=label_names,
        tokenizer=tokenizer,
        augment=True,
        augment_ratio=0.2,
        balance_classes=True,
        max_length=args.max_length,
    )
    dev_dataset = EnhancedGoEmotionsDataset(
        tsv_path=os.path.join(data_root, "goemotions_data", "dev.tsv"),
        emotion_labels=label_names,
        tokenizer=tokenizer,
        augment=False,
        max_length=args.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CustomDistilBERT(model_name="distilbert-base-uncased",
                         num_labels=len(label_names),
                         dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, optimal_thresholds = train_model(
        model, train_loader, dev_loader, device,
        epochs=args.epochs,
        early_stopping_patience=1,
        focal_loss_gamma=1.0,
        label_names=label_names,
    )

    # final eval
    model.eval()
    model.set_thresholds(optimal_thresholds)
    all_outputs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Final Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            predictions, probs = model.predict(input_ids, attention_mask)
            all_preds.append(predictions.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    micro = f1_score(all_labels, all_preds, average="micro")
    macro = f1_score(all_labels, all_preds, average="macro")
    print("\n=== Final Evaluation with Optimized Thresholds ===")
    print(f"Micro F1: {micro:.4f}")
    print(f"Macro F1: {macro:.4f}")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    print("\nOptimal thresholds for each class:")
    for label, thr in zip(label_names, optimal_thresholds):
        print(f"{label}: {thr:.3f}")


if __name__ == "__main__":
    main()
