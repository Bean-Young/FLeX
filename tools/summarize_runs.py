"""Aggregate three-run results using the manuscript's two-level protocol.

Manifest columns: setting,method,source,target,backbone,seed,result
`result` points to a run_flex.py JSON file. Values remain fractions, except HD95.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from flex.metrics import summarize_records


METRICS = ("iou", "dice", "hd95", "cls_f1", "cls_acc", "lesion_iou",
           "lesion_dice", "empty_mask_false_positive_rate", "malignant_recall")


def summarize(manifest: Path, expected_runs: int, expected_backbones: int) -> dict:
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"setting", "method", "source", "target", "backbone", "seed", "result"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")

    runs = defaultdict(list)
    for row in rows:
        result_path = Path(row["result"])
        if not result_path.is_absolute():
            result_path = manifest.parent / result_path
        result = json.loads(result_path.read_text())
        if not isinstance(result.get("records"), list) or len(result["records"]) != result.get("n"):
            raise ValueError(f"{result_path}: expected a full image-level record set")
        result = {**result, **summarize_records(result["records"])}
        key = tuple(row[field] for field in ("setting", "method", "source", "target", "backbone"))
        runs[key].append((row["seed"], result))

    backbone_rows = []
    for key, members in sorted(runs.items()):
        seeds = [seed for seed, _ in members]
        if len(seeds) != expected_runs or len(set(seeds)) != expected_runs:
            raise ValueError(f"{key}: expected {expected_runs} distinct seeds; got {seeds}")
        counts = {result["n"] for _, result in members}
        if len(counts) != 1:
            raise ValueError(f"{key}: inconsistent target image counts: {counts}")
        score = dict(zip(("setting", "method", "source", "target", "backbone"), key))
        score["n"] = counts.pop()
        score["seeds"] = seeds
        score["seed_metrics"] = {seed: {metric: result.get(metric) for metric in METRICS}
                                 for seed, result in members}
        score["metrics"] = {}
        for metric in METRICS:
            values = [result.get(metric) for _, result in members]
            if any(value is None for value in values):
                score["metrics"][metric] = None
            else:
                score["metrics"][metric] = {
                    "mean": statistics.mean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
        backbone_rows.append(score)

    domains = defaultdict(list)
    for row in backbone_rows:
        key = tuple(row[field] for field in ("setting", "method", "source", "target"))
        domains[key].append(row)
    domain_rows = []
    for key, members in sorted(domains.items()):
        if len(members) != expected_backbones:
            raise ValueError(f"{key}: expected {expected_backbones} backbones; got {len(members)}")
        counts = {row["n"] for row in members}
        if len(counts) != 1:
            raise ValueError(f"{key}: inconsistent image counts across backbones: {counts}")
        score = dict(zip(("setting", "method", "source", "target"), key))
        score["n"] = counts.pop()
        score["backbones"] = [row["backbone"] for row in members]
        seed_sets = {tuple(sorted(row["seed_metrics"])) for row in members}
        if len(seed_sets) != 1:
            raise ValueError(f"{key}: seed identifiers differ across backbones")
        seeds = sorted(members[0]["seed_metrics"])
        score["seed_metrics"] = {
            seed: {
                metric: (statistics.mean(row["seed_metrics"][seed][metric] for row in members)
                         if all(row["seed_metrics"][seed][metric] is not None for row in members) else None)
                for metric in METRICS
            }
            for seed in seeds
        }
        score["metrics"] = {
            metric: (statistics.mean(row["metrics"][metric]["mean"] for row in members)
                     if all(row["metrics"][metric] is not None for row in members) else None)
            for metric in METRICS
        }
        score["run_sd"] = {
            metric: (statistics.stdev(score["seed_metrics"][seed][metric] for seed in seeds)
                     if len(seeds) > 1 and all(score["seed_metrics"][seed][metric] is not None for seed in seeds)
                     else None)
            for metric in METRICS
        }
        domain_rows.append(score)

    source_groups = defaultdict(list)
    for row in domain_rows:
        key = tuple(row[field] for field in ("setting", "method", "source"))
        source_groups[key].append(row)
    source_rows = []
    for key, members in sorted(source_groups.items()):
        score = dict(zip(("setting", "method", "source"), key))
        score["n"] = sum(row["n"] for row in members)
        score["targets"] = [row["target"] for row in members]
        seed_sets = {tuple(sorted(row["seed_metrics"])) for row in members}
        if len(seed_sets) != 1:
            raise ValueError(f"{key}: seed identifiers differ across target domains")
        seeds = sorted(members[0]["seed_metrics"])
        score["seed_metrics"] = {
            seed: {
                metric: (sum(row["n"] * row["seed_metrics"][seed][metric] for row in members) / score["n"]
                         if score["n"] and all(row["seed_metrics"][seed][metric] is not None for row in members)
                         else None)
                for metric in METRICS
            }
            for seed in seeds
        }
        score["metrics"] = {
            metric: (sum(row["n"] * row["metrics"][metric] for row in members) / score["n"]
                     if score["n"] and all(row["metrics"][metric] is not None for row in members) else None)
            for metric in METRICS
        }
        score["run_sd"] = {
            metric: (statistics.stdev(score["seed_metrics"][seed][metric] for seed in seeds)
                     if len(seeds) > 1 and all(score["seed_metrics"][seed][metric] is not None for seed in seeds)
                     else None)
            for metric in METRICS
        }
        source_rows.append(score)
    return {"backbone_runs": backbone_rows, "domains": domain_rows, "source_means": source_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-backbones", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.manifest, args.expected_runs, args.expected_backbones)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
