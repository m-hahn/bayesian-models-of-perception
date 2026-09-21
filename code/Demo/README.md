# Behavioral Data Demo Workflow

This directory contains a self-contained workflow for running the fitting scripts on a new behavioral dataset.

The model runners, estimators, and supporting utilities in this demo come from the [`code/` directory of the companion repository of the PNAS 2026 paper](https://github.com/m-hahn/identifiability-bayesian-models/tree/main/code). They are kept in this directory so the demo can run independently of that repository.

The model code is organized as follows:

- `circular/`: circular stimulus spaces in degrees (based on `Synthetic/` in the PNAS'26 paper's repository).
- `interval/`: interval stimulus spaces on `[0, 3]` (based on `Remington/` in the PNAS'26 paper's repository).

`run_behavioral_pipeline.py` is the demo-specific wrapper. It validates a CSV, writes the three-column file expected by the scripts, selects the right runner for each requested loss, and launches that runner from the correct directory.

## Notebook: Fit Your Own Data

For an interactive workflow, open [`Fit_Your_Own_Data.ipynb`](Fit_Your_Own_Data.ipynb) in Jupyter. To use Google Colab, upload the notebook at [colab.research.google.com](https://colab.research.google.com/) or directly at https://colab.research.google.com/github/m-hahn/bayesian-models-of-perception/blob/main/code/Demo/Fit_Your_Own_Data.ipynb . The notebook walks through uploading a CSV, validating it, configuring one or more fits, reviewing cross-validation losses, and downloading a result bundle.

The notebook
and command-line calls use the same validation and execution functions, so their input rules
and generated model files are identical. In Google Colab, users can optionally
enable a GPU hardware accelerator.

It is highly recommended that you use the notebook to get acquainted with the codebase.

## Model Scripts

Circular variants:

- for your own data (or synthetic data):
  - `RunCircular_Free_CosineLoss.py`: p >= 2.
  - `RunCircular_Free_L1Loss.py`: p = 1.
  - `RunCircular_Free_L0.py`: p = 0 / MAP.
- for the original de Gardelle et al data:
  - `RunGardelle_FreePrior_CosineLoss.py`:  p >= 2.
  - `RunGardelle_FreePrior_L1Loss.py`: p=1.
  - `RunGardelle_FreePrior_ZeroTrig.py`: p = 0 / MAP.

Interval variants:

- for your own data (or synthetic data)
  - `RunInterval_Free_Lp_Round2.py`: p >= 2.
  - `RunInterval_Free_L1_Round2.py`: p = 1.
  - `RunInterval_Free_L0_Round2.py`: p = 0 / MAP.
- for the original Remington et al data:
  - `RunRemington_Free.py`: p >= 2.
  - `RunRemington_Free_FreeEncoding_L1_Round2.py`: p = 1.
  - `RunRemington_Free_Zero.py`: p = 0 / MAP.

Supporting modules from the same upstream source are stored alongside these scripts.

## External Datasets

The de Gardelle et al. and Remington et al. behavioral datasets are publicly available, but they are not included in this repository. To use the native reproduction scripts, obtain the datasets separately from the websites of the authors ([de Gardelle et al](https://sites.google.com/site/vincentdegardelle/publications) and [Remington et al](https://jazlab.org/resources/) and place them at the paths expected by the loaders:

- de Gardelle: `circular/data/GARDELLE/data.txt`
- Remington: `interval/data/REMINGTON/Datafiles/RSG_*_100.mat`

There are loaders set up for those specific data formats, but for your own data, use the CSV format described below.

## Input CSV

Use a headered CSV with these columns:

```csv
condition,stimulus,response
5,110.0,87.0
5,276.0,262.6
```

Column meanings:
- `condition`: integer condition/noise-level label.
- `stimulus`: target/stimulus value.
- `response`: observed response value.

In this version of the code, conditions are understood to denote levels of internal/sensory noise; no other condition manipulations are assumed. Other condition manipulations can be captured by appropriate extensions of the code (such as [here](https://gitlab.com/m-hahn/unifying-theory-biases/-/tree/main/code/Tomassini?ref_type=heads) for external/stimulus noise)



Circular data must use model coordinates in degrees on `[0, 360)`. Stimulus and response values must be finite and inside that range. If your experiment measures orientations modulo 180 degrees, convert each orientation `theta` to `(2 * theta) % 360` before fitting; halve circular model predictions again only when reporting them back as orientations.

Interval data must use the interval `[0, 3]`. Stimulus and response values must be finite and inside that range.

Tiny schema examples are in `input/example_circular.csv` and `input/example_interval.csv`; they are only format templates, not meaningful fitting datasets. For a ready-to-run, real-size example, use the [N = 1000 simulated circular dataset](circular/logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt). It uses the legacy three-column format and runs directly with a model script, as shown in the [Direct Import Trial](#direct-import-trial).

The notebook additionally uses the included
[N = 5000 circular dataset with five noise levels](circular/logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_8_12345_N5000_UNIFORM_STEEPPERIODIC.txt)
generated at p=8.

## Running

From this directory:

```bash
python run_behavioral_pipeline.py \
  --space circular \
  --input-csv input/my_circular_data.csv \
  --p 0 1 2 4 6 8 \
  --dataset-name my-circular-data \
  --overwrite
```

For one p = 2 circular fit:

```bash
python run_behavioral_pipeline.py \
  --space circular \
  --input-csv input/my_circular_data.csv \
  --p 2 \
  --dataset-name my-circular-data
```

For interval fits:

```bash
python run_behavioral_pipeline.py \
  --space interval \
  --input-csv input/my_interval_data.csv \
  --p 0 1 2 4 6 8 \
  --dataset-name my-interval-data \
  --overwrite
```

Useful options:

- `--grid`: override the default grid size, `180` for circular and `400` for interval. For circular fits, `180` is the number of grid cells, not a `[0, 180]` stimulus-space bound; the same is true of `_180_` in legacy circular filenames.
- `--reg-weight`: regularization weight, default `10.0`.
- `--fold`: held-out fold, default `0`.
- `--device cpu` or `--device cuda`: passed as `BIAS_MODEL_DEVICE`.
- `--plot-every`: write a circular fit figure every N fitting iterations, default `1000`; use `0` to disable.
- `--quiet`: hide the model runner's iteration-by-iteration output while retaining the wrapper's commands, output paths, and final NLL.
- `--wrap-circular`: wrap arbitrary circular stimulus/response values modulo 360 before fitting. The equivalent endpoint `360` is accepted automatically without this option.
- `--dry-run`: write the legacy input and print commands without fitting.

## Python API

The same workflow can be called from Python or another notebook. Validation is side-effect free:

```python
from run_behavioral_pipeline import (
    plot_errors_by_condition,
    run_pipeline,
    validate_csv,
)

summary = validate_csv("input/my_circular_data.csv", "circular")
print(summary)

result = run_pipeline(
    "input/my_circular_data.csv",
    "circular",
    p=[2],
    fold=0,
    reg_weight=10.0,
)
print(result.fits[0].cross_validation_loss)

figure, axes = plot_errors_by_condition(
    "input/my_circular_data.csv",
    "circular",
)
```

`run_pipeline` returns a `PipelineResult` containing the validated dataset summary, converted model-input path, and one `FitResult` per requested p value. Each fit result includes its p value, fold, regularization weight, grid size, cross-validation loss, and paths to its loss file, parameter log, and optional diagnostic figure.

## Outputs

The wrapper writes converted input files to:

- `circular/logs/SIMULATED_REPLICATE/`
- `interval/logs/SIMULATED_REPLICATE/`

The model scripts write losses and fitted parameters to:

- `circular/losses/` and `circular/logs/CROSSVALID/`
- `interval/losses/Interval/` and `interval/logs/CROSSVALID/`

Each parameter log records `condition_ids` in sorted order. The entries in `sigma_logit` use that same order, so condition labels do not need to be valid tensor indices.

The circular and interval CSV fitting scripts also write the latest diagnostic figure to
`circular/figures/` or `interval/figures/` during fitting. Each new checkpoint
replaces the preceding figure for that fit. Each diagnostic PDF has a matching
PNG preview, which the notebook displays inline. Circular figure generation can be
disabled with `--plot-every 0`.

After each fit, the wrapper prints the held-out cross-validation negative
log-likelihood (NLL) and the full paths to its NLL loss file, parameter log, and
diagnostic PDF (when enabled). The NLL is also the first line of the generated
loss file. Lower NLL values are better when comparing fits evaluated on the
same observations and fold.

The runner's optimization output is verbose by design; look for the final
`Cross-validation NLL:` line printed by the wrapper for the compact result.
Most legacy runners assert if their expected log or loss file already exists. Move or delete the corresponding generated output before rerunning the exact same command.

## Demo: Fit on Orientation data from de Gardelle et al.

The de Gardelle et al. circular dataset is not included. It is publicly available; after obtaining it, place the data file at `circular/data/GARDELLE/data.txt`.

Run the reproduction from `Demo/circular`:

```bash
cd Demo/circular
python RunGardelle_FreePrior_CosineLoss.py 8 0 10.0 180
```

This writes:

- `circular/logs/CROSSVALID/RunGardelle_FreePrior_CosineLoss.py_8_0_10.0_180.txt`
- `circular/losses/RunGardelle_FreePrior_CosineLoss.py_8_0_10.0_180.txt.txt`

A reference CPU run saved:

- training loss: `3.888923406600952`
- cross-validation loss: `992.94140625`
- saved checkpoint iteration: `13500`
- minimum checkpointed cross-validation loss: `991.9083251953125` at iteration `1500`

Analogous runs work for p>2 (replace "8" by p). At p=0 and p=1:

```bash
cd Demo/circular
python RunGardelle_FreePrior_ZeroTrig.py 0 0 10.0 180
python RunGardelle_FreePrior_L1Loss.py 1 0 10.0 180
```


## Demo: Fit on time interval data from Remington et al.

The  Remington et al. interval-data runner is included as `interval/RunRemington_Free.py`.

The Remington et al. behavioral files are not included. They are publicly available; after obtaining them, place them under:

```text
interval/data/REMINGTON/Datafiles/
```

The loader expects files matching `RSG_*_100.mat`, with `sample`, `response`, `gain`, and `correct` fields. We focus on the Ready-Set-Go gain=1 trials. 

Run the p = 2, fold-0 reproduction from `Demo/interval`:

```bash
cd Demo/interval
python RunRemington_Free.py 2 0 0.1 200
```

This writes:

- `interval/logs/CROSSVALID/RunRemington_Free.py_2_0_0.1_200.txt`
- `interval/losses/RunRemington_Free.py_2_0_0.1_200.txt.txt`

A reference CPU run saved:

- training loss: `2.755476713180542`
- cross-validation loss: `2741.94140625`
- saved checkpoint iteration: `4500`
- minimum checkpointed cross-validation loss: `2741.94140625` at iteration `4500`

Analogous runs work for p>2 (replace "2" by p). At p = 0 and p = 1, run:

```bash
cd Demo/interval
python RunRemington_Free_Zero.py 0 0 0.1 200
python RunRemington_Free_FreeEncoding_L1_Round2.py 1 0 0.1 200
```


## Direct Import Trial

Run the circular p = 2 runner from `Demo/circular` on:

`logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt`

Command:

```bash
cd Demo/circular
python RunCircular_Free_CosineLoss.py \
  2 0 10.0 180 \
  SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt
```

The NLL loss file, parameter log, and
latest diagnostic PDF are written to `losses/`, `logs/CROSSVALID/`, and
`figures/`, respectively. Their filenames begin with
`RunCircular_Free_CosineLoss.py_` and include the input filename and fit
settings; the first line of the loss file is the held-out cross-validation NLL.

A reference CPU run completed normally with a loss of approximately `44.99`;
small numerical differences can occur across library versions and hardware.

## Questions

Please do not hesitate at all to contact Michael Hahn at
[mhahn@lst.uni-saarland.de](mailto:mhahn@lst.uni-saarland.de) with any
questions. He is very happy to provide advice or help troubleshoot. You can
also open a repository issue.
