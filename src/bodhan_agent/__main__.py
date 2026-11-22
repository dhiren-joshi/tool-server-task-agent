"""Command-line entry point for the Bodhan task agent."""

from __future__ import annotations

import argparse
import json
import math
import sys

from bodhan_agent.runner import AgentRunner
from bodhan_core.errors import BodhanError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bodhan Indic-language task agent")
    parser.add_argument("--task", required=True, help="Plain-language task to complete")
    parser.add_argument("--files", required=True, help="Input file or directory")
    parser.add_argument("--out", required=True, help="Output directory for this task run")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-cost-inr", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be at least 1")
    if not math.isfinite(args.max_cost_inr) or args.max_cost_inr < 0:
        raise SystemExit("--max-cost-inr must be a finite, non-negative number")

    try:
        result = AgentRunner(
            task=args.task,
            files_path=args.files,
            output_path=args.out,
            max_steps=args.max_steps,
            max_cost_inr=args.max_cost_inr,
        ).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] not in {"completed", "unsupported"}:
            raise SystemExit(1)
    except BodhanError as exc:
        print(f"Bodhan task-agent error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
