# Code

## Overview

- [`Demo/`](Demo/) contains the runnable workflow for fitting circular or
  interval behavioral data. It includes a command-line wrapper,
  an interactive notebook, example CSV schemas, and the model scripts and data
  needed for the included reproductions.
- [`Estimators/`](Estimators/) contains the estimator implementations used by
  the modeling code.
- The Python files in this directory are utilities for loading fitted models,
  collecting dependencies, and summarizing legacy cross-validation results.

The code base is derived from [Hahn & Wei (2024)](https://gitlab.com/m-hahn/unifying-theory-biases)
and uses the same utility scripts.

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
format only; use a substantive dataset for fitting.

## Questions

Please do not hesitate to contact Michael Hahn at
[mhahn@lst.uni-saarland.de](mailto:mhahn@lst.uni-saarland.de) with any
questions. He is very happy to provide advice or help troubleshoot issues.
