import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import GoEmotionsDataset


def main():
    # Path to your data (adjust if needed)
    data_file = os.path.join("data", "train.tsv")
    label_file = os.path.join("data", "emotions.txt")

    # Load label list
    with open(label_file, "r", encoding="utf-8") as f:
        label_list = [line.strip() for line in f]

    # Load dataset
    dataset = GoEmotionsDataset(file_path=data_file, label_list=label_list)

    print(f"Dataset size: {len(dataset)} samples")
    text, labels = dataset[0]
    print("Sample text:", text[:100], "...")
    print("Sample labels:", labels)

if __name__ == "__main__":
    main()
