"""Recompute paper metrics from saved image-level FLeX predictions.

Example: python tools/audit_results.py results/*.json --output results/audit.json
The input JSON files must contain the `records` emitted by run_flex.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flex.metrics import summarize_records


def audit_file(path: Path) -> dict:
    saved = json.loads(path.read_text())
    if not isinstance(saved.get("records"), list):
        raise ValueError(f"{path}: missing image-level records")
    recomputed = summarize_records(saved["records"])
    differences = {}
    for key, value in recomputed.items():
        old = saved.get(key)
        if isinstance(value, (float, int)) and isinstance(old, (float, int)):
            if abs(float(value) - float(old)) > 1e-9:
                differences[key] = {"saved": old, "recomputed": value}
    return {"path": str(path), "recomputed": recomputed, "differences": differences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = [audit_file(path) for path in args.results]
    rendered = json.dumps(audit, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
