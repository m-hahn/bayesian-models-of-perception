#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple


ROOT = Path(__file__).resolve().parent


CONFIG = {
    "circular": {
        "basis_dir": ROOT / "circular",
        "default_grid": 180,
        "default_condition": 5,
        "stimulus_min": 0.0,
        "stimulus_max": 360.0,
        "stimulus_include_max": False,
        "filename_template": "UserBehavior_{stem}_180_2_{conditions}_N{n}.txt",
        "loss_dir": "losses",
        "writes_figures": True,
        "scripts": {
            "map": "RunSynthetic_FreePrior_ZeroTrig_OnSim.py",
            "l1": "RunSynthetic_FreePrior_L1Loss_OnSim.py",
            "lp": "RunSynthetic_FreePrior_CosineLoss_OnSim.py",
        },
    },
    "interval": {
        "basis_dir": ROOT / "interval",
        "default_grid": 400,
        "default_condition": 4,
        "stimulus_min": 0.0,
        "stimulus_max": 3.0,
        "stimulus_include_max": True,
        "filename_template": "UserBehavior_{stem}_400_2_{conditions}_N{n}_UNIFORM_UNIFORM.txt",
        "loss_dir": "losses/Interval",
        "writes_figures": False,
        "scripts": {
            "map": "RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py",
            "l1": "RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_Round2.py",
            "lp": "RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_Round2.py",
        },
    },
}


@dataclass(frozen=True)
class DatasetSummary:
    """Validated properties of an input behavioral CSV."""

    input_csv: Path
    space: str
    row_count: int
    conditions: Tuple[int, ...]
    stimulus_range: Tuple[float, float]
    response_range: Tuple[float, float]


@dataclass(frozen=True)
class FitResult:
    """Files and headline metric produced for one loss exponent."""

    p: int
    cross_validation_loss: Optional[float]
    loss_path: Path
    parameter_log_path: Path
    figure_path: Optional[Path]


@dataclass(frozen=True)
class PipelineResult:
    """Structured result returned by :func:`run_pipeline`."""

    dataset: DatasetSummary
    legacy_dataset_path: Path
    fits: Tuple[FitResult, ...]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the circular or interval fitting scripts on behavioral CSV data."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--space", required=True, choices=sorted(CONFIG))
    parser.add_argument("--p", required=True, type=int, nargs="+", help="Loss exponent(s), e.g. 0 1 2 4 6 8.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--reg-weight", type=float, default=10.0)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--dataset-name")
    parser.add_argument("--condition-column", default="condition")
    parser.add_argument("--stimulus-column", default="stimulus")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--default-condition", type=int)
    parser.add_argument("--wrap-circular", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=os.environ.get("BIAS_MODEL_DEVICE", "cpu"))
    parser.add_argument("--plot-every", type=int, default=1000, help="Write a circular fit figure every N iterations; use 0 to disable.")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def sanitized_stem(args):
    raw = args.dataset_name if args.dataset_name else args.input_csv.stem
    stem = re.sub(r"[^A-Za-z0-9.-]+", "-", raw).strip(".-")
    return stem or "dataset"


def parse_int_condition(value, line_number):
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: condition must be numeric, got {value!r}.") from exc
    if not numeric.is_integer():
        raise ValueError(f"Line {line_number}: condition must be an integer, got {value!r}.")
    return int(numeric)


def parse_float(value, column, line_number):
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: {column} must be numeric, got {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Line {line_number}: {column} must be finite, got {value!r}.")
    return numeric


def warn_if_not_spanning_stimulus_space(args, config, rows):
    low = config["stimulus_min"]
    high = config["stimulus_max"]
    tolerance = 0.01 * (high - low)
    stimuli = [stimulus for _, stimulus, _ in rows]
    minimum = min(stimuli)
    maximum = max(stimuli)
    right_bracket = "]" if config["stimulus_include_max"] else ")"
    if minimum > low + tolerance or maximum < high - tolerance:
        warnings.warn(
            f"{args.space} stimuli span [{minimum:.6g}, {maximum:.6g}], which does not cover "
            f"the declared stimulus space [{low}, {high}{right_bracket}.",
            stacklevel=2,
        )


def warn_if_circular_values_look_axial(args, rows):
    if args.space != "circular":
        return
    stimuli = [stimulus for _, stimulus, _ in rows]
    responses = [response for _, _, response in rows]
    if max(stimuli) <= 180 and max(responses) <= 180:
        warnings.warn(
            "circular stimuli and responses are all <= 180. The circular model expects "
            "directional coordinates in [0, 360), not undoubled axial/orientation "
            "coordinates in [0, 180). If these are orientations modulo 180, convert "
            "theta to (2 * theta) % 360 before fitting.",
            stacklevel=2,
        )


def load_rows(args, config):
    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    default_condition = (
        args.default_condition if args.default_condition is not None else config["default_condition"]
    )
    rows = []
    with input_csv.open(newline="") as in_file:
        reader = csv.DictReader(in_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV must have a header row.")
        missing = [
            column
            for column in [args.stimulus_column, args.response_column]
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}.")
        has_condition = args.condition_column in reader.fieldnames

        for line_number, row in enumerate(reader, start=2):
            raw_condition = row.get(args.condition_column, "") if has_condition else ""
            if raw_condition is None or raw_condition.strip() == "":
                condition = default_condition
            else:
                condition = parse_int_condition(raw_condition.strip(), line_number)

            stimulus = parse_float(row[args.stimulus_column], args.stimulus_column, line_number)
            response = parse_float(row[args.response_column], args.response_column, line_number)

            if args.space == "circular":
                if args.wrap_circular:
                    stimulus = stimulus % 360
                    response = response % 360
                if not (0 <= stimulus < 360 and 0 <= response < 360):
                    raise ValueError(
                        f"Line {line_number}: circular stimulus/response must be in [0, 360)."
                    )
            else:
                if not (0 <= stimulus <= 3 and 0 <= response <= 3):
                    raise ValueError(
                        f"Line {line_number}: interval stimulus/response must be in [0, 3]."
                    )

            rows.append((condition, stimulus, response))

    if not rows:
        raise ValueError("Input CSV has no data rows.")

    warn_if_not_spanning_stimulus_space(args, config, rows)
    warn_if_circular_values_look_axial(args, rows)
    return rows


def summarize_rows(input_csv, space, rows):
    stimuli = [stimulus for _, stimulus, _ in rows]
    responses = [response for _, _, response in rows]
    return DatasetSummary(
        input_csv=Path(input_csv).resolve(),
        space=space,
        row_count=len(rows),
        conditions=tuple(sorted({condition for condition, _, _ in rows})),
        stimulus_range=(min(stimuli), max(stimuli)),
        response_range=(min(responses), max(responses)),
    )


def _make_options(
    input_csv,
    space,
    *,
    p=(2,),
    fold=0,
    reg_weight=10.0,
    grid=None,
    dataset_name=None,
    condition_column="condition",
    stimulus_column="stimulus",
    response_column="response",
    default_condition=None,
    wrap_circular=False,
    overwrite=False,
    dry_run=False,
    device="cpu",
    plot_every=1000,
    python_executable=None,
):
    if space not in CONFIG:
        raise ValueError(f"space must be one of {sorted(CONFIG)}; got {space!r}.")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {device!r}.")
    p_values = (p,) if isinstance(p, int) else tuple(p)
    if not p_values:
        raise ValueError("p must contain at least one loss exponent.")
    for value in p_values:
        variant_for_p(space, value)
    return SimpleNamespace(
        input_csv=Path(input_csv),
        space=space,
        p=p_values,
        fold=fold,
        reg_weight=reg_weight,
        grid=grid,
        dataset_name=dataset_name,
        condition_column=condition_column,
        stimulus_column=stimulus_column,
        response_column=response_column,
        default_condition=default_condition,
        wrap_circular=wrap_circular,
        overwrite=overwrite,
        dry_run=dry_run,
        device=device,
        plot_every=plot_every,
        python=python_executable or sys.executable,
    )


def validate_csv(
    input_csv,
    space,
    *,
    condition_column="condition",
    stimulus_column="stimulus",
    response_column="response",
    default_condition=None,
    wrap_circular=False,
):
    """Validate an input CSV without writing files or starting a fit."""

    args = _make_options(
        input_csv,
        space,
        condition_column=condition_column,
        stimulus_column=stimulus_column,
        response_column=response_column,
        default_condition=default_condition,
        wrap_circular=wrap_circular,
    )
    rows = load_rows(args, CONFIG[space])
    return summarize_rows(args.input_csv, space, rows)


def variant_for_p(space, p):
    if p == 0:
        return "map"
    if p == 1:
        return "l1"
    if p >= 2:
        return "lp"
    raise ValueError(f"p must be 0, 1, or >= 2; got {p}.")


def ensure_dirs(basis_dir):
    for relative in [
        "logs/SIMULATED_REPLICATE",
        "logs/CROSSVALID",
        "losses",
        "losses/Interval",
        "figures",
    ]:
        (basis_dir / relative).mkdir(parents=True, exist_ok=True)


def write_legacy_dataset(args, config, rows):
    basis_dir = config["basis_dir"]
    ensure_dirs(basis_dir)

    condition_ids = sorted({row[0] for row in rows})
    conditions = "-".join(str(condition) for condition in condition_ids)
    filename = config["filename_template"].format(
        stem=sanitized_stem(args), conditions=conditions, n=len(rows)
    )
    out_path = basis_dir / "logs" / "SIMULATED_REPLICATE" / filename
    with out_path.open("w") as out_file:
        print("# Generated by Demo/run_behavioral_pipeline.py", file=out_file)
        print(f"# source_csv\t{args.input_csv.resolve()}", file=out_file)
        print("# columns\tcondition stimulus response", file=out_file)
        print("=======", file=out_file)
        for condition, stimulus, response in rows:
            print(f"{condition} {stimulus:.12g} {response:.12g}", file=out_file)
    return filename, out_path


def expected_outputs(config, script, fit_name, p, fold, reg_weight, grid):
    basis_dir = config["basis_dir"]
    suffix = f"{script}_{fit_name}_{p}_{fold}_{reg_weight}_{grid}.txt"
    loss_path = basis_dir / config["loss_dir"] / f"{suffix}.txt"
    log_path = basis_dir / "logs" / "CROSSVALID" / suffix
    figure_path = None
    if config.get("writes_figures"):
        figure_path = basis_dir / "figures" / f"{script}_{fit_name}_{p}_{fold}_{reg_weight}_{grid}.pdf"
    return loss_path, log_path, figure_path


def run_variant(args, config, fit_name, p):
    basis_dir = config["basis_dir"]
    grid = args.grid if args.grid is not None else config["default_grid"]
    reg_weight = str(args.reg_weight)
    variant = variant_for_p(args.space, p)
    script = config["scripts"][variant]
    loss_path, log_path, figure_path = expected_outputs(config, script, fit_name, p, args.fold, reg_weight, grid)

    if args.overwrite:
        for path in [loss_path, log_path, figure_path]:
            if path is None:
                continue
            if path.exists():
                path.unlink()
    elif loss_path.exists() or log_path.exists():
        raise FileExistsError(
            f"Output already exists for p={p}. Use --overwrite to rerun: {loss_path} / {log_path}"
        )

    command = [
        args.python,
        script,
        str(p),
        str(args.fold),
        reg_weight,
        str(grid),
        fit_name,
    ]
    print("Running:", " ".join(command), flush=True)
    print("Working directory:", basis_dir, flush=True)
    if args.dry_run:
        print("Dry run: command not executed.", flush=True)
        return loss_path, log_path, figure_path

    env = os.environ.copy()
    env["BIAS_MODEL_DEVICE"] = args.device
    env["BIAS_MODEL_PLOT_EVERY"] = str(args.plot_every)
    mpl_config = ROOT / ".matplotlib"
    mpl_config.mkdir(exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_config))

    subprocess.run(command, cwd=basis_dir, env=env, check=True)
    return loss_path, log_path, figure_path


def read_cross_validation_loss(loss_path):
    with loss_path.open() as loss_file:
        first_line = loss_file.readline().strip()
    if not first_line:
        raise ValueError(f"Loss file is empty: {loss_path}")
    try:
        return float(first_line)
    except ValueError as exc:
        raise ValueError(f"Loss file does not start with a number: {loss_path}") from exc


def _execute_pipeline(args):
    config = CONFIG[args.space]
    rows = load_rows(args, config)
    dataset = summarize_rows(args.input_csv, args.space, rows)
    fit_name, dataset_path = write_legacy_dataset(args, config, rows)
    print(f"Wrote legacy input: {dataset_path}")

    fits = []
    for p in args.p:
        loss_path, log_path, figure_path = run_variant(args, config, fit_name, p)
        print(f"Expected loss file: {loss_path}")
        print(f"Expected parameter log: {log_path}")
        if figure_path is not None:
            print(f"Expected figure: {figure_path}")
        cross_validation_loss = None if args.dry_run else read_cross_validation_loss(loss_path)
        fits.append(
            FitResult(
                p=p,
                cross_validation_loss=cross_validation_loss,
                loss_path=loss_path,
                parameter_log_path=log_path,
                figure_path=figure_path,
            )
        )
    return PipelineResult(dataset=dataset, legacy_dataset_path=dataset_path, fits=tuple(fits))


def run_pipeline(
    input_csv,
    space,
    *,
    p=(2,),
    fold=0,
    reg_weight=10.0,
    grid=None,
    dataset_name=None,
    condition_column="condition",
    stimulus_column="stimulus",
    response_column="response",
    default_condition=None,
    wrap_circular=False,
    overwrite=False,
    dry_run=False,
    device="cpu",
    plot_every=1000,
    python_executable=None,
):
    """Validate a CSV, run the requested fits, and return structured results."""

    args = _make_options(
        input_csv,
        space,
        p=p,
        fold=fold,
        reg_weight=reg_weight,
        grid=grid,
        dataset_name=dataset_name,
        condition_column=condition_column,
        stimulus_column=stimulus_column,
        response_column=response_column,
        default_condition=default_condition,
        wrap_circular=wrap_circular,
        overwrite=overwrite,
        dry_run=dry_run,
        device=device,
        plot_every=plot_every,
        python_executable=python_executable,
    )
    return _execute_pipeline(args)


def main():
    args = parse_args()
    _execute_pipeline(args)


if __name__ == "__main__":
    main()
