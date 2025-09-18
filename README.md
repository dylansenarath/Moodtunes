# MoodTunes 🎵
An AI-driven emotion-aware music recommendation system.

MoodTunes classifies emotions in text (e.g., social media posts, messages, comments) and maps them to songs with matching emotional tone, using GoEmotions for emotion detection and Emotions4MIDI for music recommendation.

---

## Quickstart

# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training (example)
python src/moodtunes_train.py ^
  --data-root data ^
  --epochs 5 ^
  --batch-size 32

---

## Project Presentation
For system architecture and example outputs, see the  
[MoodTunes Presentation (PDF)](assets/MoodTunes_Presentation.pdf).

---

## Repo Layout

src/
  dataset.py           # GoEmotions dataset loader
  model.py             # DistilBERT wrapper (CustomDistilBERT)
  moodtunes_train.py   # Training script (CLI)
tests/
  test_dataset.py      # Minimal dataset smoke test
assets/
  MoodTunes_Presentation.pdf
requirements.txt
README.md
LICENSE

---

## Dataset Setup
This repo expects local files under data/ (not tracked by git):

data/
  emotions.txt
  train.tsv
  dev.tsv
  test.tsv

> These are the GoEmotions splits you used in class. Do not commit raw data.

---

## CPU/GPU Notes
- The code auto-selects device: cuda if available, else cpu.
- On Windows with Python 3.13, PyTorch currently installs CPU-only. That’s fine for smoke tests.
- If you want GPU support, create a Python 3.11/3.12 venv and install a CUDA-enabled PyTorch build.

---

## Quick Smoke Test
Once your data is in data/:

# Run a short 1-epoch test
python -m src.moodtunes_train --data-root data --epochs 1 --batch-size 8

---

## Next Steps
- Extend tests/ with unit tests for model training loop.
- Add example evaluation results (classification reports, F1 scores).
- Integrate Emotions4MIDI for full music recommendation pipeline.
