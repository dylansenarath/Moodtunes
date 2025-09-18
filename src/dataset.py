import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import nltk

# Ensure tokenizer data is available
nltk.download("punkt", quiet=True)

class GoEmotionsDataset(Dataset):
    """
    Custom Dataset for GoEmotions.
    Each row: text, labels (comma-separated indices), id.
    Labels are converted into multi-hot vectors.
    """

    def __init__(self, file_path, label_list, augment=False, balance=False):
        self.data = pd.read_csv(file_path, sep="\t", header=None,
                                names=["text", "labels", "id"])
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.augment = augment
        self.balance = balance

        # Convert labels to multi-hot
        self.data["multi_hot"] = self.data["labels"].apply(self._labels_to_multi_hot)

        # Balance if requested
        if self.balance:
            self._apply_balancing()

    def _labels_to_multi_hot(self, label_str):
        indices = [int(x) for x in label_str.split(",") if x.strip().isdigit()]
        vec = np.zeros(self.num_labels, dtype=np.float32)
        vec[indices] = 1.0
        return vec

    def _apply_balancing(self):
        # Naive oversampling of minority classes
        class_counts = np.sum(np.stack(self.data["multi_hot"].values), axis=0)
        max_count = int(class_counts.max())

        balanced_rows = []
        for idx, row in self.data.iterrows():
            labels = np.where(row["multi_hot"] == 1)[0]
            # Oversample rows with rare labels
            oversample_factor = max_count // max(1, min(class_counts[labels]))
            balanced_rows.extend([row] * oversample_factor)

        self.data = pd.DataFrame(balanced_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        labels = torch.tensor(row["multi_hot"], dtype=torch.float32)

        if self.augment:
            text = self._simple_augment(text)

        return text, labels

    def _simple_augment(self, text):
        """Simple augmentation: random word duplication."""
        words = nltk.word_tokenize(text)
        if len(words) > 3 and random.random() < 0.3:
            pos = random.randint(0, len(words) - 1)
            words.insert(pos, words[pos])
        return " ".join(words)
