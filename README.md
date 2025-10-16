# MoodTunes 🎵
An AI-driven emotion-aware music recommendation system.

MoodTunes classifies emotions in text (e.g., social media posts, messages, comments) and maps them to songs with matching emotional tone, using GoEmotions for emotion detection and Emotions4MIDI for music recommendation.

---

## Quickstart

### 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run training (example)
python src/moodtunes_train.py ^
  --data-root data ^
  --epochs 5 ^
  --batch-size 32

---

## Project Presentation
For system architecture and example outputs, see the  
[MoodTunes Presentation (PDF)](assets/Moodtunes_Presentation.pdf).

---

## Project Paper
For a full writeup of methods, experiments, and results, see the  
[MoodTunes Paper (PDF)](assets/Moodtunes_paper.pdf).

---

## Quick Smoke Test
Once your data is in data/:

### Run a short 1-epoch test
python -m src.moodtunes_train --data-root data --epochs 1 --batch-size 8

---

## Next Steps
- Extend tests/ with unit tests for model training loop.
- Add example evaluation results (classification reports, F1 scores).
- Integrate Emotions4MIDI for full music recommendation pipeline.
