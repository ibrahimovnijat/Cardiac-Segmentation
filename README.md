# Cardiac-Segmentation

2D Left-ventricular segmentation of 2D transthoracic echocardiographic (ECG) images. Dataset contains 2D ECG sequences of 2 and 4 chamber views. 

[Link](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html) to the challenge & dataset.

## Layout

- `src/` — dataset, model, training, evaluation, and volume/EF code
- `notebooks/` — exploration, inference, and evaluation notebooks (see these for results)
- `runs/` — training outputs (checkpoints, metrics); gitignored

## Usage

Train from the repo root:

```
python src/train.py
```

Notebooks under `notebooks/` expect the CAMUS dataset one directory above the repo root (`../CAMUS_public`) and add `src/` to `sys.path` automatically.