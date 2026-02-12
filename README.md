# CSCI316 Project 1 (Dockerized)

This folder contains a PySpark workflow (`main.py`) that loads the dataset, performs feature engineering, trains/evaluates models, and saves plots to the `out/` directory.

## Prerequisites

- Docker (recommended for reproducible runs)

## Files

- `main.py`: main PySpark script
- `Mrhe_Land_Grants.csv`: dataset
- `Dockerfile`: container build instructions
- `requirements.txt`: Python dependencies
- `.github/workflows/docker.yml`: GitHub Actions workflow to build/run and upload outputs

## Run with Docker (local)

From this folder:

```powershell
docker build -t project1_final:latest .
docker run --rm -v "${PWD}\out:/app/out" project1_final:latest
```

This will create output files under `.\out\`. Output files contain graphs/visualisations used in the program (i.e. Seaborn output)

## View the plots

After the container finishes, open the PNGs in `out/`:

- `out/ml_analysis_results.png`
- `out/feature_correlation_heatmap.png`
- `out/confusion_matrix_heatmap.png`

PowerShell quick open:

```powershell
start .\out\ml_analysis_results.png
```

## Run with GitHub Actions

1. Push this folder as a GitHub repository.
2. Go to the repo’s **Actions** tab and run the workflow (or push a commit to trigger it).
3. Download the `out` artifact from the workflow run to get the generated PNGs. Download link will be generated under the "Upload outputs" section in the workflow.

