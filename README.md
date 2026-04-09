# RobustSHAP

Robust SHAP experiments and notebooks.

## Project Files

- `golub_kernel.ipynb` is the script for the Golub data set.
- `simulation_simple.ipynb` is the simulation script.

## Environment Setup (Conda)

This project includes an environment definition in `environment.yml`.

### 1. Install Conda

Use either Miniconda or Anaconda.

### 2. Create the environment from YAML

Run from the project root:

```bash
conda env create -f environment.yml
```
cs
### 3. Activate the environment

```bash
conda activate emotion-env
```

### 4. Verify installation

```bash
python --version
python -c "import numpy, pandas, sklearn, torch; print('Environment OK')"
```

### 5. (Optional) Register Jupyter kernel

```bash
python -m ipykernel install --user --name emotion-env --display-name "Python (emotion-env)"
```

Then select `Python (emotion-env)` in Jupyter notebooks.

## Updating the environment

If `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

## Notes

- The current `environment.yml` includes a `prefix:` entry tied to one local machine path.
- If environment creation fails on another machine, remove the `prefix:` line and run the create command again.
