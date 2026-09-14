# Roe-flux PINNs with residual-adaptive refinement

This repository contains the self-contained six-arm Roe-flux PINN ablation,
the archived experiment outputs, and the scripts that regenerate the figures
for the associated book chapter on wet-bed dam-break flow. The included archive
lets you recreate the quantitative figures without retraining the models.

## Contents

The repository keeps the final, reproducible materials together.

- `paperspace_rar_ablation.py` trains the six residual-and-sampling conditions
  across three random seeds.
- `runs.tar.gz` is the archived 18-run campaign used by the figure-analysis
  script. It contains predictions, adaptive-refinement anchors, metrics, and
  saved model checkpoints.
- `chapter/v1/analyze_focused_archive.py` reads `runs.tar.gz` and regenerates
  the eight quantitative figures in `chapter/v1/focused_figures/`.
- `chapter/v1/` also contains the remaining illustration and chapter-build
  utilities, the chapter Markdown source, and the generated figure assets.

## Set up the environment

Use Python 3.11 or later. Install the figure and document dependencies in an
isolated environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch build that matches the
machine before running the ablation. Regenerating figures from `runs.tar.gz`
does not require a GPU.

## Regenerate the chapter figures

The archive is already at the repository root, so run the analysis script from
the repository root. The command overwrites the quantitative PNG files in
`chapter/v1/focused_figures/` and writes `focused_analysis.json`.

```powershell
python chapter/v1/analyze_focused_archive.py
```

## Run the ablation

The full campaign requires an NVIDIA CUDA GPU. It uses six conditions, three
seeds, 16,000 initial collocation points, five refinement rounds, and a final
L-BFGS stage.

```powershell
python paperspace_rar_ablation.py --output runs/ablation --workers 1
```

To make a fresh experiment available to the figure script, archive the output
directory as `runs.tar.gz` at the repository root.

```powershell
tar -czf runs.tar.gz runs
python chapter/v1/analyze_focused_archive.py
```

Run `python paperspace_rar_ablation.py --help` to see the resume, quick-test,
worker, and residual-indicator options.

## Build the formatted chapter document

The chapter build additionally requires Pandoc, Microsoft Word on Windows, and
the optional document dependencies in `requirements.txt`.

```powershell
python chapter/v1/build_focused_chapter.py
```

This creates a DOCX, PDF, and pagination report beside the chapter source.

## Reproducibility scope

The repository intentionally excludes exploratory drafts, runtime logs, and
unrelated research outputs. The archived campaign is the sole quantitative
source for the current chapter figures.
