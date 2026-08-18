"""Run the sealed development-only behavioral analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from overfitting_spaces.development_analysis import (
    development_geometry, final_development_choices, load_and_validate, nested_cv,
    transform_flatten, whole_run_uncertainty, write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=Path("audit/runs"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/development-b42d5e4"))
    args = parser.parse_args()
    records, logits, integrity = load_and_validate(args.audit_root, args.raw_root)
    representations: dict[str, np.ndarray] = {}
    for name in ("raw_logits", "class_centered", "scale_normalized", "probabilities"):
        representations[name] = np.vstack([transform_flatten(logits[f"{row.run_id}/seed-{row.seed}/epoch-{row.epoch}"], name) for row in records])
    models, predictions = nested_cv(records, representations)
    selected = min((name for name in representations), key=lambda name: models[f"behavioral_{name}"]["pooled"]["mae"])
    final_choices = final_development_choices(records, selected, representations[selected])
    components = final_choices[f"behavioral_{selected}"]["components"]
    geometry = development_geometry(records, representations[selected], selected, components)
    uncertainty = whole_run_uncertainty(records, predictions[f"behavioral_{selected}"], predictions["all_nuisance"])
    write_outputs(args.output, records, integrity, models, predictions, geometry, uncertainty, final_choices)
    # A deliberately compact, directly reproducible visual summary.
    args.output.mkdir(parents=True, exist_ok=True)
    names = list(models)
    values = [models[name]["pooled"]["mae"] for name in names]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(range(len(names)), values)
    axis.set_xticks(range(len(names)), [name.replace("behavioral_", "behav.\n") for name in names], rotation=35, ha="right")
    axis.set_ylabel("outer-CV MAE for g")
    axis.set_title("Development-only nested grouped CV")
    fig.tight_layout()
    fig.savefig(args.output / "outer_cv_mae.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for group in sorted({record.group for record in records}):
        points = [record for record in records if record.group == group]
        axis.plot([point.epoch for point in points], [point.target for point in points], alpha=0.75, linewidth=1.2)
    axis.set_xlabel("epoch")
    axis.set_ylabel("g = evaluation loss - running minimum")
    axis.set_title("Development trajectories (12 seeds)")
    fig.tight_layout()
    fig.savefig(args.output / "development_g_trajectories.png", dpi=180)
    plt.close(fig)
    print(f"analysis complete: {args.output}")


if __name__ == "__main__":
    main()
