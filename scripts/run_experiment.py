"""Run one experiment config end to end and write its report.

    python scripts/run_experiment.py --config configs/synthetic.yaml --out results/synthetic
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

from dcai.figures import write_figures
from dcai.pipeline import StudyConfig, evaluate, synthetic_inputs
from dcai.report import render_markdown, tripwire


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-boot", type=int, help="override the config's resample count")
    args = parser.parse_args(argv)

    cfg = StudyConfig.from_yaml(args.config)
    if args.n_boot:
        cfg = dataclasses.replace(cfg, n_boot=args.n_boot)
    if cfg.synthetic is None:
        print("real-data loaders are not implemented yet (blocked on data access)",
              file=sys.stderr)
        return 2

    start = time.time()
    internal, external = synthetic_inputs(cfg)
    result = evaluate(internal, external, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    figures = write_figures(result, args.out)
    report = args.out / "report.md"
    report.write_text(render_markdown(result, synthetic=True, figures=figures))

    print(f"wrote {report} in {time.time() - start:.0f}s")
    flags = tripwire(result)
    for flag in flags:
        print(f"TRIPWIRE  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
