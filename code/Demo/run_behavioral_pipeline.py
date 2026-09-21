#!/usr/bin/env python3
import ast
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
PROGRESS_PREFIX = "BIAS_MODEL_PROGRESS"


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
            "map": "RunCircular_Free_L0.py",
            "l1": "RunCircular_Free_L1Loss.py",
            "lp": "RunCircular_Free_CosineLoss.py",
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
        "writes_figures": True,
        "scripts": {
            "map": "RunInterval_Free_L0_Round2.py",
            "l1": "RunInterval_Free_L1_Round2.py",
            "lp": "RunInterval_Free_Lp_Round2.py",
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
    fold: int
    reg_weight: float
    grid: int
    cross_validation_loss: Optional[float]
    loss_path: Path
    parameter_log_path: Path
    figure_path: Optional[Path]

    @property
    def figure_preview_path(self):
        """PNG rendering of the diagnostic PDF, when a figure was produced."""

        if self.figure_path is None:
            return None
        return self.figure_path.with_suffix(".png")


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
    parser.add_argument("--quiet", action="store_true", help="Hide verbose optimization output from the model runner.")
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
                else:
                    # The upper endpoint is the same point as zero on a circle.
                    # Accept this common export convention without requiring users
                    # to opt into wrapping arbitrary out-of-range values.
                    if stimulus == 360:
                        stimulus = 0.0
                    if response == 360:
                        response = 0.0
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
    quiet=False,
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
        quiet=quiet,
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


def _run_with_progress(command, *, cwd, env, p, fold):
    """Run a quiet fit while showing one updating progress line."""

    notebook_display = None
    try:
        from IPython import get_ipython
        from IPython.display import display

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell":
            notebook_display = display
    except ImportError:
        pass

    display_handle = None
    terminal_line_started = False
    recent_output = []

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    for line in process.stdout:
        stripped = line.rstrip()
        if stripped.startswith(f"{PROGRESS_PREFIX}\t"):
            _, iteration, train_loss = stripped.split("\t", 2)
            message = (
                f"Fitting p={p}, fold={fold}: iteration {iteration}, "
                f"train loss {float(train_loss):.6g}"
            )
            if notebook_display is not None:
                content = {"text/plain": message}
                if display_handle is None:
                    display_handle = notebook_display(
                        content, raw=True, display_id=True
                    )
                else:
                    display_handle.update(content, raw=True)
            else:
                print(f"\r{message}", end="", flush=True)
                terminal_line_started = True
        else:
            recent_output.append(line)
            recent_output = recent_output[-50:]

    return_code = process.wait()
    if terminal_line_started:
        print()
    if return_code:
        if recent_output:
            print("".join(recent_output), file=sys.stderr, end="")
        raise subprocess.CalledProcessError(return_code, command)


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
        if figure_path is not None:
            preview_path = figure_path.with_suffix(".png")
            if preview_path.exists():
                preview_path.unlink()
    elif loss_path.exists() or log_path.exists():
        raise FileExistsError(
            f"Output already exists for p={p}. Use --overwrite to rerun: {loss_path} / {log_path}"
        )

    if args.space == "circular" and args.plot_every == 0:
        figure_path = None

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
    env["BIAS_MODEL_PROGRESS"] = "1" if args.quiet else "0"
    mpl_config = ROOT / ".matplotlib"
    mpl_config.mkdir(exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_config))

    if args.quiet:
        _run_with_progress(
            command,
            cwd=basis_dir,
            env=env,
            p=p,
            fold=args.fold,
        )
    else:
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


def read_fitted_profiles(fit, space):
    """Read the fitted prior, encoding allocation, and noise parameters."""

    import numpy as np

    if space not in CONFIG:
        raise ValueError(f"space must be one of {sorted(CONFIG)}; got {space!r}.")

    raw = {}
    condition_ids = ()
    for line in fit.parameter_log_path.read_text().splitlines()[2:]:
        if line.startswith("========"):
            break
        if "\t" not in line:
            continue
        name, value = line.split("\t", 1)
        name = name.strip()
        parsed = ast.literal_eval(value.strip())
        if name == "condition_ids":
            condition_ids = tuple(parsed)
        else:
            raw[name] = np.asarray(parsed, dtype=float)

    def softmax(values):
        shifted = values - np.max(values)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    prior = len(raw["prior"]) * softmax(raw["prior"])
    encoding = len(raw["volume"]) * softmax(raw["volume"])
    upper = 360.0 if space == "circular" else 3.0
    grid = np.arange(len(prior)) * upper / len(prior)
    return {
        "grid": grid,
        "upper": upper,
        "prior": prior,
        "encoding": encoding,
        "condition_ids": condition_ids,
        "raw": raw,
    }


def fitted_profile_roughness(fit, space):
    """Return mean squared adjacent changes for the prior and encoding."""

    import numpy as np

    profiles = read_fitted_profiles(fit, space)

    def one(values):
        if space == "circular":
            values = np.r_[values, values[0]]
        return float(np.mean(np.diff(values) ** 2))

    return one(profiles["prior"]), one(profiles["encoding"])


def plot_errors_by_condition(input_csv, space):
    """Plot signed response errors and their dispersion for each condition."""

    import matplotlib.pyplot as plt
    import numpy as np

    args = _make_options(
        input_csv,
        space,
        wrap_circular=(space == "circular"),
    )
    rows = load_rows(args, CONFIG[space])
    conditions = np.asarray([row[0] for row in rows])
    stimuli = np.asarray([row[1] for row in rows])
    responses = np.asarray([row[2] for row in rows])
    errors = responses - stimuli
    if space == "circular":
        errors = (errors + 180) % 360 - 180

    condition_ids = sorted(set(conditions))
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for condition in condition_ids:
        mask = conditions == condition
        axes[0].scatter(
            stimuli[mask],
            errors[mask],
            s=8,
            alpha=0.18,
            label=str(condition),
        )

    empirical_sd = [
        errors[conditions == condition].std(ddof=1)
        for condition in condition_ids
    ]
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(
        xlabel="stimulus (degrees)" if space == "circular" else "stimulus",
        ylabel="signed response error",
        title="Errors by condition",
    )
    axes[0].legend(title="condition", frameon=False, ncol=2)
    axes[1].plot(condition_ids, empirical_sd, marker="o")
    axes[1].set(
        xticks=condition_ids,
        xlabel="condition",
        ylabel="empirical error SD",
        title="Error dispersion by condition",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure, axes


def plot_loss_function_nll(fits, reference_p, title=None):
    """Plot delta NLL in the style of the PNAS paper's Figure 4 code."""

    import matplotlib.pyplot as plt

    if reference_p not in fits:
        raise ValueError(f"reference_p={reference_p} is not present in fits.")

    p_values = sorted(fits)
    reference_nll = fits[reference_p].cross_validation_loss
    delta_nll = [fits[p].cross_validation_loss - reference_nll for p in p_values]

    figure, axis = plt.subplots(figsize=(4.2, 3.6), layout="constrained")
    axis.plot(p_values, delta_nll, color="gray", linewidth=1)
    axis.scatter(p_values, delta_nll, color="gray", zorder=3)
    axis.axhline(0, color="gray", linestyle="dotted", linewidth=1)
    axis.set(
        xlabel="Exponent",
        ylabel=r"$\Delta$ NLL",
        title=title,
        xticks=p_values,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    ymin, ymax = axis.get_ylim()
    axis.vlines(
        reference_p,
        ymin,
        ymax,
        color="gray",
        linestyle="dotted",
        linewidth=1,
    )
    axis.set_ylim(ymin, ymax)
    return figure, axis


def plot_cross_validated_loss_function_nll(fits_by_p, title=None):
    """Plot mean paired delta NLL with standard errors across folds."""

    import matplotlib.pyplot as plt
    import numpy as np

    if not fits_by_p:
        raise ValueError("fits_by_p must contain at least one loss exponent.")

    p_values = sorted(fits_by_p)
    losses_by_p = {}
    fold_ids = None
    for p in p_values:
        losses_for_p = {fit.fold: fit.cross_validation_loss for fit in fits_by_p[p]}
        if any(value is None for value in losses_for_p.values()):
            raise ValueError(f"Missing cross-validation NLL for p={p}.")
        if fold_ids is None:
            fold_ids = sorted(losses_for_p)
        elif sorted(losses_for_p) != fold_ids:
            raise ValueError("Every loss exponent must have results for the same folds.")
        losses_by_p[p] = np.asarray([losses_for_p[fold] for fold in fold_ids])

    if len(fold_ids) < 2:
        raise ValueError("At least two folds are required for cross-fold error bars.")

    reference_p = min(p_values, key=lambda p: losses_by_p[p].mean())
    paired_deltas = np.stack(
        [losses_by_p[p] - losses_by_p[reference_p] for p in p_values]
    )
    mean_deltas = paired_deltas.mean(axis=1)
    standard_errors = paired_deltas.std(axis=1, ddof=0) / math.sqrt(len(fold_ids))

    figure, axis = plt.subplots(figsize=(4.2, 3.6), layout="constrained")
    axis.plot(p_values, mean_deltas, color="gray", linewidth=0.75)
    axis.scatter(p_values, mean_deltas, color="gray", s=18, zorder=3)
    axis.errorbar(
        p_values,
        mean_deltas,
        yerr=standard_errors,
        color="gray",
        fmt="none",
        linewidth=0.75,
        capsize=3,
    )
    axis.axhline(0, color="gray", linestyle="dotted", linewidth=1)
    axis.set(
        xlabel="Exponent",
        ylabel=r"$\Delta$ NLL",
        title=title,
        xticks=p_values,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure, axis


def _prepare_included_example(source_name, output_name):
    source_path = ROOT / "circular" / "logs" / "SIMULATED_REPLICATE" / source_name
    output_dir = ROOT / "input" / "uploads"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    body = source_path.read_text().split("=======\n", 1)[1].strip().splitlines()
    with output_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["condition", "stimulus", "response"])
        writer.writerows(line.split() for line in body)
    return output_path


def prepare_included_example():
    """Prepare the N=1000 circular example as a CSV and return its path."""

    return _prepare_included_example(
        "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_"
        "180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt",
        "tutorial_simulated_circular.csv",
    )


def prepare_five_condition_example():
    """Prepare the N=5000 five-condition circular example as a CSV."""

    return _prepare_included_example(
        "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_"
        "180_8_12345_N5000_UNIFORM_STEEPPERIODIC.txt",
        "tutorial_simulated_p8_five_noise_levels.csv",
    )


def _execute_pipeline(args):
    config = CONFIG[args.space]
    rows = load_rows(args, config)
    dataset = summarize_rows(args.input_csv, args.space, rows)
    fit_name, dataset_path = write_legacy_dataset(args, config, rows)
    print(f"Wrote model input: {dataset_path}")

    fits = []
    for p in args.p:
        loss_path, log_path, figure_path = run_variant(args, config, fit_name, p)
        output_label = "Expected" if args.dry_run else "Output"
        print(f"{output_label} NLL loss file: {loss_path}")
        print(f"{output_label} parameter log: {log_path}")
        if figure_path is not None:
            print(f"{output_label} diagnostic figure: {figure_path}")
        cross_validation_loss = None if args.dry_run else read_cross_validation_loss(loss_path)
        if cross_validation_loss is not None:
            print(f"Cross-validation NLL: {cross_validation_loss}")
        fits.append(
            FitResult(
                p=p,
                fold=args.fold,
                reg_weight=args.reg_weight,
                grid=args.grid if args.grid is not None else config["default_grid"],
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
    quiet=False,
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
        quiet=quiet,
        python_executable=python_executable,
    )
    return _execute_pipeline(args)


def main():
    args = parse_args()
    _execute_pipeline(args)


if __name__ == "__main__":
    main()
