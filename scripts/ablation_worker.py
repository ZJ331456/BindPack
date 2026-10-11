#!/usr/bin/env python3
"""Run an experimental adapter through the normal dataset evaluator."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bindpack.experiments.online import worker_class
import eval_dataset


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ablation-mode', dest='mode', choices=(
        'capture', 'no_plan_prune', 'binding_string', 'binding_relation', 'qd', 'iterative'), required=True)
    parser.add_argument('--snapshot-dir', required=True)
    parser.add_argument('--ledger-path', required=True)
    parser.add_argument('--budget-fraction', type=float, default=1.0)
    options, rest = parser.parse_known_args()
    args = eval_dataset.build_parser().parse_args(rest)
    if (not args.reuse_index or args.embedding_batch_size != 4 or args.llm_prefetch_workers != 8
            or args.max_new_tokens != 2048 or args.eval_mode != 'retrieve'):
        raise ValueError('Ablations require frozen index, batch4, workers8, max_tokens2048, retrieve only')
    options.selected_indices = json.loads(Path(args.sample_indices_file).read_text())
    eval_dataset.BindPack = worker_class(options)
    eval_dataset.run_eval(args.dataset, args)


if __name__ == '__main__':
    main()
