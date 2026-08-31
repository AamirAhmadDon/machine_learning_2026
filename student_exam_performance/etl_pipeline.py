import argparse
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ETLPipelineError(ValueError):
    """Custom exception for validation failures in the ETL pipeline."""


def _candidate_csv_path(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "student_exam_performance.csv"),
        os.path.join(base_dir, "student_exam_performance (3).csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No input CSV file found. Expected 'student_exam_performance.csv' or 'student_exam_performance (3).csv'."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for input, config, database, and chart paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Student exam performance ETL pipeline")
    parser.add_argument("--input", default=_candidate_csv_path(base_dir), help="Path to the raw CSV dataset")
    parser.add_argument("--config", default=os.path.join(base_dir, "column_metadata.json"), help="Schema and validation config JSON path")
    parser.add_argument("--db", default=os.path.join(base_dir, "student_performance.db"), help="SQLite output database path")
    parser.add_argument("--charts-dir", default=os.path.join(base_dir, "charts"), help="Directory for generated charts")
    parser.add_argument("--summary-report", default=os.path.join(base_dir, "etl_summary_report.txt"), help="Path for ETL summary report")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return parser.parse_args()


def load_config(config_path: str = "column_metadata.json") -> Dict[str, Any]:
    """Load ETL schema configuration and create it if it does not exist."""
    default_config: Dict[str, Any] = {
        "numeric_bounds": {
            "age": [0, 100],
            "previous_exam_score": [0, 100],
            "previous_gpa": [0.0, 4.0],
            "attendance_percentage": [0.0, 100.0],
            "assignment_completion_rate": [0.0, 100.0],
            "study_hours_per_day": [0.0, 24.0],
            "self_study_hours": [0.0, 24.0],
            "private_tuition": [0, 10],
            "online_learning_hours": [0.0, 24.0],
            "sleep_hours": [0.0, 24.0],
            "daily_screen_time": [0.0, 24.0],
            "physical_activity_hours": [0.0, 24.0],
            "stress_level": [0, 10],
            "questions_attempted": [0, 100],
            "questions_correct": [0, 100],
            "time_management_score": [0.0, 10.0],
            "exam_anxiety_level": [0, 10],
            "exam_score": [0.0, 100.0],
            "accuracy_rate": [0.0, 1.0],
        },
        "categorical_values": {
            "gender": ["Female", "Male", "Other"],
            "education_level": [
                "High School",
                "Undergraduate",
                "Graduate",
                "Postgraduate",
                "Primary",
                "Middle School",
            ],
            "school_type": ["Public", "Private", "Charter", "International"],
            "family_income": ["Low", "Middle", "High"],
            "urban_rural": ["Urban", "Suburban", "Rural"],
            "parent_education": [
                "High School",
                "Bachelor",
                "Master",
                "Doctorate",
                "Unknown",
            ],
            "study_environment": ["Quiet", "Moderate", "Noisy", "Unknown"],
            "study_method": [
                "Flashcards",
                "Summarizing",
                "Self-Reading",
                "Practice Tests",
                "Group Study",
                "Active Recall",
                "Concept Maps",
                "Unknown",
            ],
            "revision_frequency": ["Rarely", "Weekly", "Daily", "Occasionally", "Unknown"],
            "study_consistency": ["Low", "Medium", "High", "Unknown"],
            "notes_quality": ["Poor", "Average", "Good", "Excellent", "Unknown"],
            "sleep_quality": ["Poor", "Fair", "Average", "Good", "Excellent", "Unknown"],
            "device_availability": ["Dedicated", "Shared", "Unknown"],
            "internet_access": ["Yes", "No", "Unknown"],
            "performance_grade": ["A", "B", "C", "D", "F", "Unknown"],
            "pass_status": ["Pass", "Fail", "Unknown"],
            "performance_level": ["High", "Medium", "Low", "Unknown"],
            "exam_difficulty": ["Easy", "Medium", "Hard", "Unknown"],
        },
    }

    config_path = os.path.abspath(config_path)
    if os.path.exists(config_path):
        logger.info("Loading configuration from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        merged = default_config.copy()
        for key, value in loaded.items():
            if isinstance(value, dict):
                merged[key] = {**default_config.get(key, {}), **value}
            else:
                merged[key] = value
        return merged

    logger.info("Configuration file not found. Creating default config at %s", config_path)
    with open(config_path, "w", encoding="utf-8") as file_handle:
        json.dump(default_config, file_handle, indent=2)
    return default_config


def extract_data(file_path: str) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame and log a summary."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    logger.info("Reading dataset from %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Dataset loaded: %s rows, %s columns", df.shape[0], df.shape[1])
    logger.info("Columns: %s", ", ".join(df.columns.tolist()))
    logger.info("Missing values summary:\n%s", df.isna().sum().sort_values(ascending=False).head(15).to_string())
    print(f"Initial rows: {df.shape[0]}")
    print(f"Initial columns: {df.shape[1]}")
    print(df.head(3).to_string(index=False))
    return df


def validate_schema(df: pd.DataFrame, config: dict) -> None:
    """Validate the data has the expected required columns and no bad numeric rows remain."""
    required_columns = [
        "student_id",
        "gender",
        "school_type",
        "attendance_percentage",
        "previous_gpa",
        "exam_score",
        "performance_grade",
        "pass_status",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ETLPipelineError(f"Missing required columns for ETL processing: {', '.join(missing)}")

    if "questions_attempted" in df.columns and df["questions_attempted"].isna().any():
        bad_rows = df.index[df["questions_attempted"].isna()].tolist()[:5]
        raise ETLPipelineError(f"Bad row data detected: questions_attempted is null at indices {bad_rows}")

    if "questions_correct" in df.columns and df["questions_correct"].isna().any():
        bad_rows = df.index[df["questions_correct"].isna()].tolist()[:5]
        raise ETLPipelineError(f"Bad row data detected: questions_correct is null at indices {bad_rows}")

    if "exam_score" in df.columns and df["exam_score"].isna().any():
        bad_rows = df.index[df["exam_score"].isna()].tolist()[:5]
        raise ETLPipelineError(f"Bad row data detected: exam_score is null at indices {bad_rows}")

    numeric_bounds = config.get("numeric_bounds", {})
    for column, (lower_bound, upper_bound) in numeric_bounds.items():
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.isna().sum() > 0:
                bad_rows = df.index[values.isna()].tolist()[:5]
                raise ETLPipelineError(f"Bad row data detected: {column} has non-numeric values or nulls at indices {bad_rows}")


def transform_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean and enrich the dataset for downstream processing."""
    cleaned = df.copy()

    numeric_bounds = config.get("numeric_bounds", {})
    categorical_values = config.get("categorical_values", {})

    valid_numeric_columns: list[str] = []
    for column in numeric_bounds:
        if column in cleaned.columns:
            converted = pd.to_numeric(cleaned[column], errors="coerce")
            if converted.notna().sum() > 0:
                cleaned[column] = converted
                valid_numeric_columns.append(column)
            else:
                logger.info("Skipping numeric bounds validation for non-numeric column: %s", column)

    for column in ["previous_gpa", "attendance_percentage", "time_management_score"]:
        if column in cleaned.columns:
            median_value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(median_value)

    for column in ["parent_education", "notes_quality", "sleep_quality", "device_availability"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown").astype(str).replace({"nan": "Unknown", "None": "Unknown"})

    for column, (lower_bound, upper_bound) in numeric_bounds.items():
        if column in cleaned.columns and column in valid_numeric_columns:
            cleaned[column] = cleaned[column].clip(lower=float(lower_bound), upper=float(upper_bound))

    if {"questions_correct", "questions_attempted"}.issubset(cleaned.columns):
        cleaned["questions_attempted"] = pd.to_numeric(cleaned["questions_attempted"], errors="coerce").fillna(0)
        cleaned["questions_correct"] = pd.to_numeric(cleaned["questions_correct"], errors="coerce").fillna(0)
        cleaned.loc[cleaned["questions_correct"] > cleaned["questions_attempted"], "questions_correct"] = cleaned.loc[
            cleaned["questions_correct"] > cleaned["questions_attempted"], "questions_attempted"
        ]
        cleaned["questions_attempted"] = cleaned["questions_attempted"].clip(lower=0)
        cleaned["questions_correct"] = cleaned["questions_correct"].clip(lower=0)

    if {"questions_correct", "questions_attempted"}.issubset(cleaned.columns):
        cleaned["accuracy_rate"] = np.where(
            cleaned["questions_attempted"] > 0,
            (cleaned["questions_correct"] / cleaned["questions_attempted"]).round(4),
            0.0,
        )

    if {"study_hours_per_day", "self_study_hours", "online_learning_hours"}.issubset(cleaned.columns):
        cleaned["total_study_hours"] = (
            cleaned["study_hours_per_day"].fillna(0)
            + cleaned["self_study_hours"].fillna(0)
            + cleaned["online_learning_hours"].fillna(0)
        )

    if {"attendance_percentage", "previous_gpa"}.issubset(cleaned.columns):
        cleaned["is_high_risk"] = (
            (cleaned["attendance_percentage"] < 75) & (cleaned["previous_gpa"] < 2.5)
        ).astype(int)

    for column, allowed_values in categorical_values.items():
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip().replace({"nan": "Unknown", "None": "Unknown"})
            valid_values = set(str(value) for value in allowed_values)
            cleaned.loc[~cleaned[column].isin(valid_values), column] = "Unknown"

    logger.info("Dataset cleaning completed. Final shape: %s rows, %s columns", cleaned.shape[0], cleaned.shape[1])
    validate_schema(cleaned, config)
    logger.info("Schema validation passed for cleaned dataset.")
    return cleaned


def generate_charts(df: pd.DataFrame, output_dir: str) -> None:
    """Generate static charts for documentation and reporting."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating chart outputs in %s", output_dir)

    grade_path = os.path.join(output_dir, "grade_distribution.png")
    if "performance_grade" in df.columns:
        grade_counts = df["performance_grade"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        grade_counts.plot(kind="bar", color="steelblue", edgecolor="black", ax=ax)
        ax.set_title("Performance Grade Distribution")
        ax.set_xlabel("Performance Grade")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(grade_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    difficulty_path = os.path.join(output_dir, "score_vs_difficulty.png")
    if {"exam_score", "exam_difficulty"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="exam_difficulty", y="exam_score", order=sorted(df["exam_difficulty"].dropna().unique()), ax=ax)
        ax.set_title("Exam Score vs Difficulty")
        ax.set_xlabel("Exam Difficulty")
        ax.set_ylabel("Exam Score")
        fig.tight_layout()
        fig.savefig(difficulty_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    numeric_focus = [
        "study_hours_per_day",
        "self_study_hours",
        "online_learning_hours",
        "total_study_hours",
        "previous_gpa",
        "attendance_percentage",
        "time_management_score",
        "accuracy_rate",
        "exam_score",
    ]
    available_numeric = [column for column in numeric_focus if column in df.columns]
    if len(available_numeric) > 1:
        corr_df = df[available_numeric].astype(float)
        corr = corr_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title("Study Metrics Correlation Matrix")
        fig.tight_layout()
        fig.savefig(corr_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved chart files: %s, %s, %s", grade_path, difficulty_path, corr_path)


def load_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """Write the cleaned dataset to SQLite using sqlite3 and SQLAlchemy engine."""
    logger.info("Loading cleaned data into SQLite database: %s", db_path)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql("clean_student_performance", con=engine, if_exists="replace", index=False)
        row_count = pd.read_sql_query("SELECT COUNT(*) AS total_rows FROM clean_student_performance", connection).iloc[0, 0]
        logger.info("SQLite table created successfully with %s rows", row_count)
        print(f"SQLite table created with {row_count} rows.")


def run_verification_queries(db_path: str) -> Dict[str, List[tuple]]:
    """Run and print inspection queries against the SQLite table."""
    logger.info("Running verification queries against %s", db_path)
    conn = sqlite3.connect(db_path)
    results: Dict[str, List[tuple]] = {}

    try:
        query_a = """
            SELECT
                performance_grade,
                AVG(exam_score) AS avg_exam_score,
                AVG(accuracy_rate) AS avg_accuracy_rate,
                COUNT(*) AS total_count
            FROM clean_student_performance
            GROUP BY performance_grade
            ORDER BY performance_grade;
        """
        query_b = """
            SELECT
                study_method,
                AVG(exam_score) AS avg_exam_score
            FROM clean_student_performance
            WHERE attendance_percentage >= 80
            GROUP BY study_method
            ORDER BY avg_exam_score DESC
            LIMIT 5;
        """
        query_c = """
            SELECT
                pass_status,
                SUM(is_high_risk) AS high_risk_count
            FROM clean_student_performance
            GROUP BY pass_status
            ORDER BY pass_status;
        """

        results["query_a"] = conn.execute(query_a).fetchall()
        results["query_b"] = conn.execute(query_b).fetchall()
        results["query_c"] = conn.execute(query_c).fetchall()

        print("\nQuery A: Average exam score, accuracy rate, and count by performance grade")
        print(pd.DataFrame(results["query_a"], columns=["performance_grade", "avg_exam_score", "avg_accuracy_rate", "total_count"]).to_string(index=False))

        print("\nQuery B: Top 5 study methods by average exam score for attendance >= 80")
        print(pd.DataFrame(results["query_b"], columns=["study_method", "avg_exam_score"]).to_string(index=False))

        print("\nQuery C: Count of high-risk students grouped by pass_status")
        print(pd.DataFrame(results["query_c"], columns=["pass_status", "high_risk_count"]).to_string(index=False))
        return results
    finally:
        conn.close()


def write_summary_report(report_path: str, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, db_path: str, charts_dir: str, query_results: Dict[str, List[tuple]]) -> None:
    """Write a concise ETL summary report for stakeholders and operational checks."""
    try:
        query_a_df = pd.DataFrame(query_results["query_a"], columns=["performance_grade", "avg_exam_score", "avg_accuracy_rate", "total_count"])
        query_b_df = pd.DataFrame(query_results["query_b"], columns=["study_method", "avg_exam_score"])
        query_c_df = pd.DataFrame(query_results["query_c"], columns=["pass_status", "high_risk_count"])
    except KeyError:
        query_a_df = pd.DataFrame()
        query_b_df = pd.DataFrame()
        query_c_df = pd.DataFrame()

    metrics = {
        "input_rows": int(raw_df.shape[0]),
        "input_columns": int(raw_df.shape[1]),
        "cleaned_rows": int(cleaned_df.shape[0]),
        "cleaned_columns": int(cleaned_df.shape[1]),
        "missing_values_after_clean": int(cleaned_df.isna().sum().sum()),
        "db_path": db_path,
        "database_rows": int(pd.read_sql_query("SELECT COUNT(*) AS total_rows FROM clean_student_performance", sqlite3.connect(db_path)).iloc[0, 0]),
        "charts": sorted([name for name in os.listdir(charts_dir) if name.endswith(".png")]) if os.path.isdir(charts_dir) else [],
        "avg_exam_score_by_grade": query_a_df.to_dict(orient="records"),
        "top_study_methods_by_score": query_b_df.to_dict(orient="records"),
        "risk_by_pass_status": query_c_df.to_dict(orient="records"),
    }

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("Student Exam Performance ETL Summary\n")
        report_file.write("===================================\n\n")
        report_file.write(f"Input rows: {metrics['input_rows']}\n")
        report_file.write(f"Input columns: {metrics['input_columns']}\n")
        report_file.write(f"Cleaned rows: {metrics['cleaned_rows']}\n")
        report_file.write(f"Cleaned columns: {metrics['cleaned_columns']}\n")
        report_file.write(f"Missing values remaining after cleaning: {metrics['missing_values_after_clean']}\n")
        report_file.write(f"Database rows: {metrics['database_rows']}\n")
        report_file.write(f"Charts generated: {', '.join(metrics['charts']) if metrics['charts'] else 'none'}\n\n")
        report_file.write("Performance grade summary:\n")
        for row in metrics["avg_exam_score_by_grade"]:
            report_file.write(f"  - {row['performance_grade']}: avg_exam_score={row['avg_exam_score']:.2f}, avg_accuracy_rate={row['avg_accuracy_rate']:.4f}, total_count={row['total_count']}\n")
        report_file.write("\nTop study methods by score:\n")
        for row in metrics["top_study_methods_by_score"]:
            report_file.write(f"  - {row['study_method']}: avg_exam_score={row['avg_exam_score']:.2f}\n")
        report_file.write("\nHigh-risk count by pass status:\n")
        for row in metrics["risk_by_pass_status"]:
            report_file.write(f"  - {row['pass_status']}: high_risk_count={int(row['high_risk_count'])}\n")

    logger.info("ETL summary report written to %s", report_path)


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.input
    db_path = args.db
    config_path = args.config
    charts_dir = args.charts_dir
    summary_report_path = args.summary_report
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    start_time = time.perf_counter()
    logger.info("Starting ETL pipeline...")
    logger.info("Input CSV: %s", csv_path)
    logger.info("Output DB: %s", db_path)
    logger.info("Charts directory: %s", charts_dir)

    try:
        config = load_config(config_path)
        raw_df = extract_data(csv_path)
        cleaned_df = transform_data(raw_df, config)
        generate_charts(cleaned_df, charts_dir)
        load_to_sqlite(cleaned_df, db_path)
        query_results = run_verification_queries(db_path)
        write_summary_report(summary_report_path, raw_df, cleaned_df, db_path, charts_dir, query_results)
    except ETLPipelineError as exc:
        logger.exception("ETL pipeline failed due to data validation issue: %s", exc)
        raise
    except Exception as exc:
        logger.exception("ETL pipeline failed: %s", exc)
        raise
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info("ETL pipeline completed in %.2f seconds", elapsed)
        print(f"\nETL pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
