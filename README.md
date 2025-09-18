## Quickstart
```bash
# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training (example)
python src/moodtunes_train.py ^
  --data-root path\to\project\goemotions_data ^
  --epochs 5 ^
  --batch-size 32
