## JAX Implementations of Diffusion Policy

This repository provides unofficial reimplementations of existing diffusion policies for offline-RL in JAX.

### Getting started:
```
conda env create --name jax_metarl python==3.11.8
pip3 install -e . # should install this repo and dependencies
```

### Example command 

```
python3 main.py
```

Also supports using Ray for hyperparameter search and WandB for logging experiment metrics. Use `smoke_test` to toggle Ray tune. 

### File organization:


Diffusion Policy
- [ ]