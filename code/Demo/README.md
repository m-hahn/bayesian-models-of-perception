# Behavioral Data Demo Workflow

This directory contains a self-contained workflow for running the fitting scripts on a new behavioral dataset.

The model runners, estimators, and supporting utilities in this demo come from the [`code/` directory of the companion repository of the PNAS 2026 paper](https://github.com/m-hahn/identifiability-bayesian-models/tree/main/code). They are kept in this directory so the demo can run independently of that repository.

The model code is organized as follows:

- `circular/`: circular stimulus spaces in degrees, based on `Synthetic/`.
- `interval/`: interval stimulus spaces on `[0, 3]`, based on `Remington/`.

`run_behavioral_pipeline.py` is the demo-specific wrapper. It validates a CSV, writes the three-column file expected by the scripts, selects the right runner for each requested loss, and launches that runner from the correct directory.

## Notebook: Fit Your Own Data

For an interactive workflow, open [`Fit_Your_Own_Data.ipynb`](Fit_Your_Own_Data.ipynb) in Jupyter. To use Google Colab, upload the notebook at [colab.research.google.com](https://colab.research.google.com/). The notebook walks through uploading a CSV, validating it, configuring one or more fits, reviewing cross-validation losses, and downloading a result bundle.

Start with one p = 2 fit. Running multiple loss exponents or folds can take considerably longer. The notebook and CLI call the same validation and execution functions, so their input rules and generated model files are identical.

## Model Scripts

Circular variants:

- for your own data (or synthetic data):
  - `RunCircular_Free_CosineLoss.py`: p >= 2.
  - `RunCircular_Free_L1Loss.py`: p = 1.
  - `RunCircular_Free_L0.py`: p = 0 / MAP.
- for the original de Gardelle et al data:
  - `RunGardelle_FreePrior_CosineLoss.py`:  p >= 2.
  - `RunGardelle_FreePrior_L1Loss_Downsampled_TargetSize.py`: p=1.
  - `RunGardelle_FreePrior_ZeroTrig_Downsampled_TargetSize.py`: p = 0 / MAP.

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

Interval data must use the Remington scale `[0, 3]`. Stimulus and response values must be finite and inside that range.

Tiny schema examples are in `input/example_circular.csv` and `input/example_interval.csv`; they are only format templates, not meaningful fitting datasets. For a ready-to-run, real-size example, use the [N = 1000 simulated circular dataset](circular/logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt). It uses the legacy three-column format and runs directly with a model script, as shown in the [Direct Import Trial](#direct-import-trial).

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
- `--wrap-circular`: wrap circular stimulus/response values modulo 360 before fitting.
- `--dry-run`: write the legacy input and print commands without fitting.

## Python API

The same workflow can be called from Python or another notebook. Validation is side-effect free:

```python
from run_behavioral_pipeline import run_pipeline, validate_csv

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
```

`run_pipeline` returns a `PipelineResult` containing the validated dataset summary, converted legacy dataset path, and one `FitResult` per requested p value. Each fit result includes the parsed cross-validation loss and paths to its loss file, parameter log, and optional circular diagnostic figure.

## Outputs

The wrapper writes converted input files to:

- `circular/logs/SIMULATED_REPLICATE/`
- `interval/logs/SIMULATED_REPLICATE/`

The model scripts write losses and fitted parameters to:

- `circular/losses/` and `circular/logs/CROSSVALID/`
- `interval/losses/Interval/` and `interval/logs/CROSSVALID/`

Each parameter log records `condition_ids` in sorted order. The entries in `sigma_logit` use that same order, so condition labels do not need to be valid tensor indices.

The circular fitting scripts also write the latest fit diagnostic figure to `circular/figures/` during fitting.

The stdout from fitting is verbose by design; the compact result to compare across p values is the cross-validation loss in the generated `losses` file.
Most legacy runners assert if their expected log or loss file already exists. Move or delete the corresponding generated output before rerunning the exact same command.

## Demo: Reproduce de Gardelle et al.

The de Gardelle et al. circular dataset is not included. It is publicly available; after obtaining it, place the data file at `circular/data/GARDELLE/data.txt`. This dataset uses the native loader rather than the CSV wrapper above. The raw de Gardelle orientation columns are modulo 180 degrees; the loader converts them into model coordinates by doubling `rot` and `resp_rot` before fitting. It then checks that stimuli and responses are finite and in the circular stimulus space `[0, 360)`, and warns if the observed stimuli do not span that full space within a 1% edge tolerance.

Run the reproduction from `Demo/circular`:

```bash
cd Demo/circular
python RunGardelle_FreePrior_CosineLoss.py 8 0 10.0 180
```

This writes:

- `circular/logs/CROSSVALID/RunGardelle_FreePrior_CosineLoss.py_8_0_10.0_180.txt`
- `circular/losses/RunGardelle_FreePrior_CosineLoss.py_8_0_10.0_180.txt.txt`

A reference CPU run completed normally and saved:

- training loss: `3.888923406600952`
- cross-validation loss: `992.94140625`
- saved checkpoint iteration: `13500`
- minimum checkpointed cross-validation loss: `991.9083251953125` at iteration `1500`

The upstream repository's saved log for the same command reports training loss `3.8888871669769287`, cross-validation loss `992.8677978515625`, and the same minimum-CV checkpoint at iteration `1500`. The runner, estimator, and utility code in this demo are byte-identical to their upstream counterparts. The externally obtained `data/GARDELLE/data.txt` used for the reference run was also verified against the upstream file. The included loader preserves the upstream loading logic with added bounds checks. The remaining numerical differences are expected small runtime/library drift rather than a data or implementation mismatch.

The p = 0 and p = 1 de Gardelle controls in the original repository are the downsampled target-size scripts. To run the all-levels, target-size-9936 controls that correspond to the saved old-repo outputs:

```bash
cd Demo/circular
python RunGardelle_FreePrior_ZeroTrig_Downsampled_TargetSize.py 0 0 10.0 180 1-2-3-4-5 9936 ''
python RunGardelle_FreePrior_L1Loss_Downsampled_TargetSize.py 1 0 10.0 180 1-2-3-4-5 9936 ''
```

The final empty argument selects the scripts' default seed behavior and reproduces the unseeded legacy filenames. In reference CPU runs:

- p = 0 wrote cross-validation loss `1352.499267578125` at checkpoint `1000`; the original saved output is `1352.4991455078125` at checkpoint `1000`.
- p = 1 wrote best cross-validation loss `1232.50146484375` at checkpoint `9000`; the original saved output is `1230.20068359375` at checkpoint `10000`.


## Demo: Reproduce Remington et al.

The native Remington et al. interval-data runner is included as `interval/RunRemington_Free.py`. It uses the native Remington `.mat` loader rather than the CSV wrapper above. The loader checks that stimuli are finite and in the interval stimulus space `[0, 3]`, checks responses against the same bounds, and warns if the observed stimuli do not span that full space within a 1% edge tolerance.

The Remington et al. Ready-Set-Go gain-1 behavioral files are not included. They are publicly available; after obtaining them, place them under:

```text
interval/data/REMINGTON/Datafiles/
```

The loader expects files matching `RSG_*_100.mat`, with `sample`, `response`, `gain`, and `correct` fields.

Run the p = 8, fold-0 reproduction from `Demo/interval`:

```bash
cd Demo/interval
python RunRemington_Free.py 8 0 0.1 200
```

This writes:

- `interval/logs/CROSSVALID/RunRemington_Free.py_8_0_0.1_200.txt`
- `interval/losses/RunRemington_Free.py_8_0_0.1_200.txt.txt`

A reference CPU run completed normally and saved:

- training loss: `2.755476713180542`
- cross-validation loss: `2741.94140625`
- saved checkpoint iteration: `4500`
- minimum checkpointed cross-validation loss: `2741.94140625` at iteration `4500`

The original repository's saved log for the same command reports:

- training loss: `2.755476474761963`
- cross-validation loss: `2741.933349609375`
- saved checkpoint iteration: `4500`
- minimum checkpointed cross-validation loss: `2741.933349609375` at iteration `4500`

The runner, estimator, and utility code in this demo are byte-identical to their upstream counterparts; the included loader preserves the upstream loading logic with added bounds checks. The reproduced training trajectory differs from the upstream result by at most `0.0000012` over saved checkpoints, and the cross-validation trajectory differs by at most `0.0081` summed NLL.

To reproduce the full ten-fold result for p = 8, rerun the same command with fold IDs `0` through `9`. The original repository also has saved `RunRemington_Free.py` results for p values `2`, `4`, `6`, `8`, and `10` with the same `0.1` regularization weight and `200`-point grid.

The p = 0 and p = 1 native Remington controls are:

```bash
cd Demo/interval
python RunRemington_Free_Zero.py 0 0 0.1 200
python RunRemington_Free_FreeEncoding_L1_Round2.py 1 0 0.1 200
```

In reference CPU runs:

- p = 1 completed normally and wrote cross-validation loss `2743.12255859375` at checkpoint `9500`; the original saved output is `2743.127685546875` at checkpoint `10000`.
- p = 0 was run as a bounded trajectory check through checkpoint `10500`, because the original saved p = 0 endpoint is checkpoint `539500` and takes several hours on CPU. At checkpoint `10500`, the reference run wrote cross-validation loss `2745.314453125`; the original trajectory at the same checkpoint is `2745.314208984375`.

The p = 0 command is therefore verified to run and to match the early original trajectory, but the full historical p = 0 endpoint was not rerun here.

## Direct Import Trial

The upstream-derived circular p = 2 runner was tested from `Demo/circular` on:

`logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt`

Command:

```bash
cd Demo/circular
python RunCircular_Free_CosineLoss.py \
  2 0 10.0 180 \
  SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt
```

A reference CPU run completed normally and wrote a loss value of `44.991607666015625`.
