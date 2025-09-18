\## Quickstart

```bash

\# 1. Create virtual environment

python -m venv .venv

.\\.venv\\Scripts\\activate   # Windows



\# 2. Install dependencies

pip install -r requirements.txt



\# 3. Run training (example)

python src/moodtunes\_train.py ^

&nbsp; --data-root path\\to\\project\\goemotions\_data ^

&nbsp; --epochs 5 ^

&nbsp; --batch-size 32



