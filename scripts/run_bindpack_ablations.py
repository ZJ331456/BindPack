#!/usr/bin/env python3
"""Run Full, w/o Route, w/o Plan and w/o Pack using Qwen3 models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.stage_ablations import main

if __name__ == '__main__':
    raise SystemExit(main())
