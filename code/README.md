# Code

## Overview

- [`Demo/`](Demo/) contains the runnable workflow for fitting circular or
  interval behavioral data. It includes a command-line wrapper,
  an interactive notebook, example CSV schemas, and the model scripts needed
  for the reproductions. The public de Gardelle et al. and Remington et al.
  datasets must be obtained separately, as described in the demo README.
- [`Estimators/`](Estimators/) contains the estimator implementations used by
  the modeling code.
- The Python files in this directory are utilities for loading fitted models,
  collecting dependencies, and summarizing legacy cross-validation results.

The model runners and utilities used by the demo come from the
[`code/` directory of the upstream `identifiability-bayesian-models`
repository](https://github.com/m-hahn/identifiability-bayesian-models/tree/main/code),
associated with Hahn & Wei (2024).

## Requirements

The code was developed with Python 3.9.18. The pinned dependencies are listed
in [`requirements.txt`](requirements.txt): PyTorch, Matplotlib, SciPy, and
NumPy.

From this directory, a minimal setup is:

```bash
conda create -n identifiability python=3.9 -y
conda activate identifiability
python -m pip install -r requirements.txt
```

Fitting runs on CPU by default. To use a CUDA-capable GPU, install the matching
PyTorch build and set `BIAS_MODEL_DEVICE=cuda`.

## Fit behavioral data

See [`Demo/README.md`](Demo/README.md) for the input schema, validation rules,
CLI and Python APIs, generated outputs, and instructions for reproducing the de
Gardelle et al. and Remington et al. fits.

For an interactive workflow, open
[`Demo/Fit_Your_Own_Data.ipynb`](Demo/Fit_Your_Own_Data.ipynb). To inspect the
CLI without running a fit:

```bash
cd Demo
python run_behavioral_pipeline.py --help
```

The small files in [`Demo/input/`](Demo/input/) illustrate the required CSV
format only. For a ready-to-run, real-size example, use the
[N = 1000 simulated circular dataset](Demo/circular/logs/SIMULATED_REPLICATE/SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_5_N1000_UNIFORM_STEEPPERIODIC.txt),
which uses the legacy three-column format and is documented in the demo's
[Direct Import Trial](Demo/README.md#direct-import-trial). Otherwise, provide
your own substantive CSV dataset.

## Questions

Please open a repository issue if you have questions or encounter a problem.
