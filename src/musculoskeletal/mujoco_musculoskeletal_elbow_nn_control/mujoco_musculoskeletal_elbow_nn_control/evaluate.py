"""Offline grouped evaluation for expert and neural activation controllers."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .mlp import ActivationMLP, FEATURES, TARGETS, score


def evaluate(data_dir: Path, model_path: Path):
    model = ActivationMLP.load(model_path); report = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
        x = np.asarray([[float(row[k]) for k in FEATURES] for row in rows]); y = np.asarray([[float(row[k]) for k in TARGETS] for row in rows])
        key = path.stem; metrics = score(model, x, y).__dict__; metrics["samples"] = len(rows); metrics["load_kg"] = float(rows[0]["load_kg"]); metrics["trajectory"] = "sine" if "sine" in key else "step"; report[key] = metrics
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate elbow controllers / 评价肘关节控制器")
    parser.add_argument("--data", type=Path, required=True); parser.add_argument("--model", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv); report = evaluate(args.data, args.model); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
