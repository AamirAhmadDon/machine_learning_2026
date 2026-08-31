# Student Performance ETL Pipeline

This project loads the student exam performance dataset, cleans it, creates a SQLite database, and saves chart assets for quick exploration.

## Setup

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the ETL pipeline

```bash
python etl_pipeline.py
```

The script expects the raw CSV to be in the project root as either `student_exam_performance.csv` or `student_exam_performance (3).csv`.

## Outputs

- `student_performance.db` — SQLite database with the cleaned dataset
- `charts/grade_distribution.png` — grade distribution bar chart
- `charts/score_vs_difficulty.png` — box plot of exam scores by difficulty
- `charts/correlation_matrix.png` — study metrics correlation heatmap
- `column_metadata.json` — validation and schema metadata used by the ETL process
