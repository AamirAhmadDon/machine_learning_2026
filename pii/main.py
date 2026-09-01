"""
PII Data Pipeline: ETL, Data Quality, Anomaly Detection, Analytics & SQLite
===========================================================================

A single-file, portfolio-quality data engineering pipeline that:

    * Extracts a PII / NER-style dataset from a raw CSV
    * Profiles and cleans the raw data in memory (pandas)
    * Validates emails, phones, and URLs
    * Validates token / label annotations
    * Detects exact duplicates, repeated values, and potential duplicate identities
    * Performs rule-based and ML (IsolationForest) anomaly detection
    * Computes a transparent, record-level data-quality score
    * Generates a suite of data visualizations
    * Loads the cleaned data into a normalized SQLite database
    * Runs SQL analytics against the database
    * Performs a final validation before reporting success

Everything is contained in this single module so the project remains a clean,
reproducible, and easy-to-review portfolio artifact.

Author       : Portfolio project
Dependencies : pandas, numpy, matplotlib, seaborn, scikit-learn
"""

from __future__ import annotations

import ast
import json
import logging
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    "input_file": "pii_dataset.csv",
    "database_file": "pii_database.db",
    "log_file": "etl.log",
    "random_seed": 42,
    "anomaly_contamination": 0.05,
    "visualizations_dir": "visualizations",
}

# Penalties applied to the record-level quality score. The score starts at
# 100 and each documented issue subtracts its penalty. The logic is fully
# transparent so the scoring is not a black box.
QUALITY_PENALTIES: Dict[str, float] = {
    "invalid_email": 20.0,
    "invalid_phone": 15.0,
    "invalid_url": 5.0,          # URLs may legitimately be absent
    "missing_required": 10.0,    # missing critical contact fields
    "duplicate_identity": 10.0,  # potential duplicate person
    "token_label_mismatch": 15.0,
    "anomaly_flag": 10.0,
    "invalid_labels": 5.0,
}

REQUIRED_FIELDS: List[str] = ["name", "email", "phone", "job", "address", "hobby"]

# Regex patterns used for conservative field validation.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# A phone is any sequence containing at least 6 digits (optionally with
# parentheses, dashes, spaces, plus signs and dots). Conservative on purpose.
PHONE_RE = re.compile(r"^[+()\d][\d\s()\-+.]{5,}$")
# URL must be a non-empty string with a scheme or a host-like structure.
URL_RE = re.compile(
    r"^(?:(?:https?|ftp)://)?(?:[\w-]+\.)+[a-zA-Z]{2,}(?:[/?#][^\s]*)?$"
)


# ============================================================
# LOGGING
# ============================================================


def setup_logging(log_file: str) -> logging.Logger:
    """Configure a console + file logger and return a module-level logger.

    Sensitive PII is never written to the log. Only counts, percentages and
    audit metadata are logged.
    """
    logger = logging.getLogger("pii_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as exc:
        # Never silently swallow: report to console but continue.
        logger.error("Could not create log file %r: %s", log_file, exc)

    return logger


# ============================================================
# CONFIG LOADING
# ============================================================


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file merged over defaults.

    Raises RuntimeError if the config file is not valid JSON.
    """
    cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
    path = Path(config_path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError("config.json must contain a JSON object")
            cfg.update(loaded)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Invalid configuration in {config_path}: {exc}") from exc
    return cfg


# ============================================================
# DATA EXTRACTION
# ============================================================


def load_dataset(input_file: str) -> pd.DataFrame:
    """Load the raw CSV into a pandas DataFrame.

    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    df = pd.read_csv(path)
    # Normalize column names: strip whitespace and collapse case.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_list_column(series: pd.Series) -> pd.Series:
    """Safely parse Python-serialized list columns (tokens/labels/whitespace)."""
    parsed = series.map(_safe_literal_eval)
    return parsed


def _safe_literal_eval(value: Any) -> Any:
    """Try ast.literal_eval, returning the original string on failure."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


# ============================================================
# DATA PROFILING
# ============================================================


def profile_dataset(
    df: pd.DataFrame,
    raw_missing: pd.Series,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Compute a broad profile of the raw data and print a summary.

    Returns a dict of key profile metrics used later for before/after analysis.
    """
    n_rows, n_cols = df.shape
    dup_rows = int(df.duplicated().sum())
    profile = {
        "rows": n_rows,
        "columns": n_cols,
        "duplicate_rows": dup_rows,
        "missing_total": int(raw_missing.sum().sum()),
        "column_names": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "unique_counts": {c: int(df[c].nunique(dropna=False)) for c in df.columns},
    }

    print("\n==================== RAW DATA PROFILE ====================")
    print(f"Rows: {n_rows}  |  Columns: {n_cols}")
    print(f"Exact duplicate rows: {dup_rows}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1_048_576:.2f} MB")
    print("\nColumn-wise missing values:")
    col_missing = raw_missing.sum()
    for col in df.columns:
        cnt = int(col_missing.get(col, 0))
        pct = (cnt / n_rows * 100) if n_rows else 0.0
        uniq = profile["unique_counts"][col]
        print(f"  {col:<22} missing={cnt:<6} ({pct:5.1f}%)  unique={uniq}")
    print("==========================================================\n")

    logger.info("Data profile computed: %d rows, %d columns", n_rows, n_cols)
    logger.info("Exact duplicate rows in raw data: %d", dup_rows)
    logger.info("Total missing values across all columns: %d", profile["missing_total"])
    return profile


def _detect_missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame marking which cells are 'missing'.

    Recognizes NaN/None/null, empty strings, whitespace-only strings, and the
    common textual representations N/A and NA (case-insensitive).
    """
    missing_list: List[List[bool]] = []
    for _, row in df.iterrows():
        miss = []
        for val in row:
            if pd.isna(val):
                miss.append(True)
            elif isinstance(val, str):
                s = val.strip()
                miss.append(not s or s.lower() in {"n/a", "na", "null", "none"})
            else:
                miss.append(False)
        missing_list.append(miss)
    return pd.DataFrame(missing_list, index=df.index, columns=df.columns)


# ============================================================
# DATA CLEANING
# ============================================================


def _clean_text(value: Any) -> Any:
    """Trim whitespace and collapse repeated internal whitespace."""
    if pd.isna(value) or not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def _normalize_email(value: Any) -> Tuple[Any, bool]:
    """Return (normalized_email, is_valid)."""
    if pd.isna(value) or not isinstance(value, str):
        return value, False
    email = value.strip().lower()
    if not EMAIL_RE.fullmatch(email):
        return email, False
    return email, True


def _normalize_phone(value: Any) -> Tuple[Any, bool]:
    """Return (normalized_phone, is_valid)."""
    if pd.isna(value) or not isinstance(value, str):
        return value, False
    phone = value.strip()
    if not PHONE_RE.fullmatch(phone):
        return phone, False
    # Count digits: a plausible phone has at least 6 digits and no more than 15.
    digits = re.sub(r"\D", "", phone)
    if not (6 <= len(digits) <= 15):
        return phone, False
    return phone, True


def _normalize_url(value: Any) -> Tuple[Any, bool]:
    """Return (normalized_url, is_valid)."""
    if pd.isna(value) or not isinstance(value, str):
        return value, False
    url = value.strip()
    return url, bool(URL_RE.fullmatch(url))


def clean_data(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, int, int]:
    """Clean the raw DataFrame in memory (no CSV written).

    Applies whitespace normalization to textual fields and derives normalized
    + valid variants for email, phone and URL. The original raw columns are
    preserved for reference.
    """
    out = df.copy()

    # Keep the original raw values in clearly-named reference columns for
    # before/after comparison, then replace the canonical columns with cleaned
    # versions. No column-name collisions occur.
    free_text_cols = ["document", "text", "prompt", "name", "job", "address",
                      "username", "hobby"]
    for col in free_text_cols:
        if col in out.columns:
            out[col + "_raw"] = out[col]

    # General text cleaning on free-form and structured text columns.
    text_cols = free_text_cols + ["email", "phone", "url"]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].map(_clean_text)

    # Email normalization + validation.
    out["email_normalized"], out["email_valid"] = zip(
        *out["email"].map(_normalize_email)
    )

    # Phone normalization + validation.
    out["phone_normalized"], out["phone_valid"] = zip(
        *out["phone"].map(_normalize_phone)
    )

    # URL normalization + validation.
    out["url_normalized"], out["url_valid"] = zip(
        *out["url"].map(_normalize_url)
    )

    raw_missing = _detect_missing_mask(df[["name", "email", "phone", "job",
                                           "address", "username", "url", "hobby"]])
    cleaned_missing = _detect_missing_mask(out[["name", "email", "phone", "job",
                                                "address", "username", "url", "hobby"]])

    raw_total = int(raw_missing.sum().sum())
    cleaned_total = int(cleaned_missing.sum().sum())
    logger.info(
        "Cleaning complete. Missing cells raw=%d -> cleaned=%d "
        "(normalization intentionally preserves legitimate absence).",
        raw_total, cleaned_total,
    )

    return out, raw_total, cleaned_total


# ============================================================
# TOKENS AND LABELS PARSING
# ============================================================


def parse_tokens_and_labels(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Parse serialized tokens/labels/trailing_whitespace and derive features.

    Returns the DataFrame with new columns: tokens_parsed, labels_parsed,
    token_count, label_count, token_label_match, pii_entity_count,
    pii_token_count, pii_ratio, entity_sequence, entity_labels_valid, and
    invalid_labels_flag.
    """
    out = df.copy()

    out["tokens_parsed"] = _parse_list_column(df["tokens"])
    out["labels_parsed"] = _parse_list_column(df["labels"])

    if "trailing_whitespace" in out.columns:
        out["whitespace_parsed"] = _parse_list_column(df["trailing_whitespace"])
    else:
        out["whitespace_parsed"] = np.nan

    # Derived counts.
    out["token_count"] = out["tokens_parsed"].map(
        lambda x: len(x) if isinstance(x, list) else np.nan
    )
    out["label_count"] = out["labels_parsed"].map(
        lambda x: len(x) if isinstance(x, list) else np.nan
    )
    out["token_label_match"] = (
        (out["token_count"].notna())
        & (out["label_count"].notna())
        & (out["token_count"] == out["label_count"])
    )

    # Entity analysis over the label sequences.
    results = out["labels_parsed"].map(_analyze_label_sequence)
    for key in (
        "pii_token_count", "pii_entity_count", "entity_sequence",
        "entity_labels_valid", "invalid_labels_flag", "unique_token_count",
    ):
        out[key] = results.map(lambda r: r[key])

    out["pii_ratio"] = np.where(
        out["token_count"] > 0,
        out["pii_token_count"] / out["token_count"],
        0.0,
    )

    mismatch_count = int((~out["token_label_match"]).sum())
    invalid_label_rows = int(out["invalid_labels_flag"].sum())
    logger.info(
        "Tokens/labels parsed. Token-label mismatches: %d, "
        "rows with invalid label sequences: %d.",
        mismatch_count, invalid_label_rows,
    )
    return out


def _analyze_label_sequence(labels: Any) -> Dict[str, Any]:
    """Analyze a single label list for PII counts and annotation validity."""
    base: Dict[str, Any] = {
        "pii_token_count": 0, "pii_entity_count": 0,
        "entity_sequence": [], "entity_labels_valid": True,
        "invalid_labels_flag": False, "unique_token_count": 0,
    }
    if not isinstance(labels, list):
        base["invalid_labels_flag"] = True
        base["entity_labels_valid"] = False
        return base

    entity_types = []
    pii_count = 0
    prev_type = None
    i = 0
    label_set: set = set()

    while i < len(labels):
        lab = labels[i]
        if not isinstance(lab, str):
            base["invalid_labels_flag"] = True
            base["entity_labels_valid"] = False
            i += 1
            continue
        label_set.add(lab)
        if lab == "O":
            prev_type = None
            i += 1
            continue
        if "-" in lab:
            prefix, etype = lab.split("-", 1)
        else:
            prefix, etype = "", lab

        if prefix == "B":
            # A new entity. Reject consecutive 'B' of same type as an
            # annotation hint but don't alter data - only flag.
            if etype == prev_type and prev_type is not None:
                base["entity_labels_valid"] = False
                base["invalid_labels_flag"] = True
            entity_types.append(etype)
            prev_type = etype
            pii_count += 1
            i += 1
            continue

        if prefix == "I":
            if prev_type is None:
                # 'I' without a preceding B-entity start is suspicious.
                base["entity_labels_valid"] = False
                base["invalid_labels_flag"] = True
            entity_types.append(etype)
            prev_type = etype
            pii_count += 1
            i += 1
            continue

        # Some unknown prefix -> suspicious.
        base["entity_labels_valid"] = False
        base["invalid_labels_flag"] = True
        i += 1

    # Count distinct entities (spans) - approximate by transitions into B-.
    distinct_entities = 0
    prev_prefix = None
    for lab in labels:
        if not isinstance(lab, str):
            continue
        if lab == "O":
            prev_prefix = "O"
            continue
        prefix = lab.split("-", 1)[0] if "-" in lab else ""
        if prefix == "B":
            distinct_entities += 1
        prev_prefix = prefix

    base["pii_token_count"] = pii_count
    base["pii_entity_count"] = distinct_entities
    # entity_sequence as list of (start_type) not needed; keep types list.
    base["entity_sequence"] = entity_types
    base["unique_token_count"] = 0  # resolved separately (needs tokens)
    return base


def _count_unique_tokens(tokens: Any) -> int:
    """Return number of unique tokens (case-sensitive)."""
    if not isinstance(tokens, list):
        return 0
    return len(set(str(t) for t in tokens))


# ============================================================
# DUPLICATE AND REPEATED DATA DETECTION
# ============================================================


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag exact duplicates and potential duplicate identities.

    Returns a copy of df with columns: duplicate_exact, duplicate_email,
    duplicate_phone, duplicate_identity.
    """
    out = df.copy()

    # Exact duplicate rows (based on hashable original raw columns only).
    # Exclude list/parsed columns which are unhashable.
    hashable_cols = [c for c in df.columns 
                     if c not in ["tokens", "labels", "tokens_parsed", "labels_parsed", 
                                  "whitespace_parsed", "entity_sequence", "_rule_flags"]]
    dup_mask = df[hashable_cols].duplicated(keep=False) if hashable_cols else pd.Series(False, index=df.index)
    out["duplicate_exact"] = dup_mask.astype(int)
    # In this dataset there are 0 exact duplicates; keep the flag for generality.

    # Repeated emails / phones (appearing more than once across records).
    has_email = out["email"].notna()
    has_phone = out["phone"].notna()

    email_counts = out.loc[has_email, "email"].map(_lower_str).value_counts()
    phone_counts = out.loc[has_phone, "phone"].map(_clean_str).value_counts()

    out["duplicate_email"] = 0
    out["duplicate_phone"] = 0
    out.loc[has_email, "duplicate_email"] = (
        out.loc[has_email, "email"].map(_lower_str).map(lambda x: int(email_counts.get(x, 0) > 1))
    ).fillna(0)
    out.loc[has_phone, "duplicate_phone"] = (
        out.loc[has_phone, "phone"].map(_clean_str).map(lambda x: int(phone_counts.get(x, 0) > 1))
    ).fillna(0)

    # Potential duplicate identity: same email OR same phone among multiple rows.
    out["duplicate_identity"] = (
        (out["duplicate_email"] == 1) | (out["duplicate_phone"] == 1)
    ).astype(int)
    out["duplicate_flag"] = out["duplicate_exact"].copy()
    return out


def _lower_str(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def _clean_str(v: Any) -> Any:
    return v.strip() if isinstance(v, str) else v


def analyze_repeated_values(df: pd.DataFrame) -> Dict[str, int]:
    """Count records sharing a repeated value in categorical PII-ish fields.

    Returns a dict mapping column -> number of records whose value appears in
    more than one row (legitimate repetition is NOT flagged as an error).
    """
    result: Dict[str, int] = {}
    for col in ["name", "email", "phone", "address", "job", "hobby", "username", "url"]:
        if col not in df.columns:
            continue
        counts = df[col].dropna().value_counts()
        repeated_value_rows = int(counts[counts > 1].sum())
        result[col] = repeated_value_rows
    return result


# ============================================================
# VALIDATION FLAGS
# ============================================================


def _flag_annotations(out: pd.DataFrame) -> pd.DataFrame:
    """Create explicit invalid_* and mismatch flag columns."""
    out = out.copy()
    out["invalid_email"] = (~out["email_valid"]).astype(int)
    out["invalid_phone"] = (~out["phone_valid"]).astype(int)
    out["invalid_url"] = (~out["url_valid"]).astype(int)
    out["invalid_labels"] = out["invalid_labels_flag"].astype(int)
    out["token_label_mismatch"] = (~out["token_label_match"]).astype(int)

    # Missing required fields count (critical contact fields).
    required = [c for c in REQUIRED_FIELDS if c in out.columns]
    out["missing_count"] = out[required].isna().sum(axis=1)
    return out


def _accumulate_rule_anomalies(row: Dict[str, Any], flags: List[Tuple[str, str, float]]) -> None:
    """Conservative rule-based anomaly checks. Mutates flags list."""
    text = row.get("text") or ""
    if isinstance(text, str):
        tl = len(text)
        if tl >= 1 and tl < 80:
            pass  # not anomalous in this corpus; kept simple
        if tl > 1200:
            flags.append(("long_document", f"Text length {tl} characters", 2))
    if row.get("token_label_mismatch"):
        flags.append(("token_label_mismatch", "Token/label length mismatch", 3))
    if row.get("invalid_labels"):
        flags.append(("invalid_labels", "Suspicious annotation sequence", 3))
    if row.get("pii_ratio", 0) and pd.notna(row.get("pii_ratio")):
        if row["pii_ratio"] > 0.5:
            flags.append(
                ("high_pii_density", f"PII ratio {row['pii_ratio']:.2f}", 2)
            )
    if row.get("duplicate_identity"):
        flags.append(("duplicate_identity", "Potential duplicate person record", 2))
    if row.get("invalid_email"):
        flags.append(("invalid_email", "Invalid email format", 1))
    if row.get("invalid_phone"):
        flags.append(("invalid_phone", "Invalid phone format", 1))
    if row.get("invalid_url"):
        flags.append(("invalid_url", "Invalid URL format", 1))


# ============================================================
# ANOMALY DETECTION
# ============================================================


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float,
    random_seed: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run rule-based + IsolationForest anomaly detection.

    Returns (df_with_anomaly_flags, list_of_anomaly_records).

    The rule-based pass adds explicit flags; the IsolationForest pass provides
    a statistical outlier score. A statistical outlier is NOT an error -- it is
    a record flagged for investigation.
    """
    out = df.copy()

    # Each row's list of (type, description, severity) rule-based anomalies.
    rule_flags: List[List[Tuple[str, str, float]]] = []
    for _, row in out.iterrows():
        flags: List[Tuple[str, str, float]] = []
        _accumulate_rule_anomalies(row, flags)
        rule_flags.append(flags)

    out["_rule_flags"] = rule_flags
    out["rule_anomaly_count"] = out["_rule_flags"].map(len)

    # IsolationForest on meaningful numerical features (no raw text).
    feature_cols = [
        "text_length", "token_count", "pii_token_count", "pii_entity_count",
        "pii_ratio", "unique_token_count",
    ]
    available = [c for c in feature_cols if c in out.columns]

    X = out[available].replace([np.inf, -np.inf], np.nan)
    # Fill missing feature values with the column median to keep the model fit.
    for col in available:
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        X[col] = X[col].fillna(med)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_seed,
    )
    scores = model.fit_predict(X)
    # IsolationForest -1 == outlier, 1 == inlier.
    out["anomaly_flag"] = (scores == -1).astype(int)
    # Negative 'score_samples' sortable; convert to a positive-ish anomaly score.
    raw_scores = model.score_samples(X)
    # Normalize to 0..1 where higher = more anomalous.
    lo, hi = raw_scores.min(), raw_scores.max()
    span = (hi - lo) or 1.0
    out["anomaly_score"] = (hi - raw_scores) / span

    # A record is considered an outlier if either rule-based OR model flags it.
    out["anomaly_any"] = (
        (out["anomaly_flag"] == 1) | (out["rule_anomaly_count"] > 0)
    ).astype(int)

    anomaly_records: List[Dict[str, Any]] = []

    # Rule-based records.
    for idx, flags in enumerate(rule_flags):
        if flags:
            for atype, desc, sev in flags:
                anomaly_records.append(
                    {
                        "row_index": int(out.index[idx]),
                        "document_id": out.iloc[idx].get("document"),
                        "anomaly_type": atype,
                        "description": desc,
                        "severity": sev,
                        "anomaly_score": float(out.iloc[idx].get("anomaly_score", 0.0)),
                        "source": "rule",
                    }
                )

    # ML-based records.
    for idx in out.index[out["anomaly_flag"] == 1]:
        anomaly_records.append(
            {
                "row_index": int(idx),
                "document_id": out.loc[idx, "document"],
                "anomaly_type": "statistical_outlier",
                "description": "IsolationForest outlier on derived features",
                "severity": 2,
                "anomaly_score": float(out.loc[idx, "anomaly_score"]),
                "source": "ml",
            }
        )

    logger.info(
        "Anomaly detection complete: %d statistical outliers (contamination=%.2f), "
        "%d records with rule-based flags, %d total anomaly annotations.",
        int((out["anomaly_flag"] == 1).sum()),
        contamination,
        int((out["rule_anomaly_count"] > 0).sum()),
        len(anomaly_records),
    )
    return out, anomaly_records


# ============================================================
# DERIVED FEATURES
# ============================================================


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numerical features and text-length statistics."""
    out = df.copy()
    out["text_length"] = out["text"].map(
        lambda x: len(x) if isinstance(x, str) else 0
    )
    out["unique_token_count"] = out["tokens_parsed"].map(_count_unique_tokens)
    return out


# ============================================================
# DATA QUALITY SCORING
# ============================================================


def calculate_quality_scores(out: pd.DataFrame) -> pd.DataFrame:
    """Compute a transparent record-level quality score (start 100, subtract
    documented penalties) and a quality category."""
    score = pd.Series(100.0, index=out.index)

    def apply_penalty(mask: pd.Series, penalty: float) -> pd.Series:
        return mask.astype(float) * penalty

    if "invalid_email" in out.columns:
        score -= apply_penalty(out["invalid_email"].astype(bool), QUALITY_PENALTIES["invalid_email"])
    if "invalid_phone" in out.columns:
        score -= apply_penalty(out["invalid_phone"].astype(bool), QUALITY_PENALTIES["invalid_phone"])
    if "invalid_url" in out.columns:
        score -= apply_penalty(out["invalid_url"].astype(bool), QUALITY_PENALTIES["invalid_url"])
    if "missing_count" in out.columns:
        score -= apply_penalty(out["missing_count"] > 0, QUALITY_PENALTIES["missing_required"])
    if "duplicate_identity" in out.columns:
        score -= apply_penalty(out["duplicate_identity"].astype(bool), QUALITY_PENALTIES["duplicate_identity"])
    if "token_label_mismatch" in out.columns:
        score -= apply_penalty(out["token_label_mismatch"].astype(bool), QUALITY_PENALTIES["token_label_mismatch"])
    if "anomaly_any" in out.columns:
        score -= apply_penalty(out["anomaly_any"].astype(bool), QUALITY_PENALTIES["anomaly_flag"])
    if "invalid_labels" in out.columns:
        score -= apply_penalty(out["invalid_labels"].astype(bool), QUALITY_PENALTIES["invalid_labels"])

    # Clamp to a sensible range.
    out["quality_score"] = score.clip(lower=0.0, upper=100.0)

    def categorize(s: float) -> str:
        if s >= 90:
            return "Excellent"
        if s >= 75:
            return "Good"
        if s >= 50:
            return "Needs Review"
        return "Poor"

    out["quality_category"] = out["quality_score"].map(categorize)
    return out


# ============================================================
# VISUALIZATION
# ============================================================


def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _setup_styles() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.facecolor"] = "white"


def generate_visualizations(
    df: pd.DataFrame,
    before_after: Dict[str, Any],
    viz_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate a focused suite of charts and write them to viz_dir."""
    _setup_styles()
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. PII entity distribution (bar chart).
    entity_counts: Counter = Counter()
    for seq in df["entity_sequence"]:
        if isinstance(seq, list):
            for et in seq:
                entity_counts[et] += 1
    if entity_counts:
        ec = pd.Series(entity_counts).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ec.plot(kind="bar", ax=ax, color=sns.color_palette("Blues", len(ec)))
        ax.set_title("PII Entity Type Distribution (from NER labels)")
        ax.set_xlabel("Entity type")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, viz_dir / "pii_entity_distribution.png")

    # 2. Missing values by column (bar chart).
    missing = (df[["name", "email", "phone", "job", "address",
                   "username", "url", "hobby"]].isna().sum()).sort_values(ascending=False)
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        missing.plot(kind="bar", ax=ax, color="coral")
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing count")
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, viz_dir / "missing_values.png")

    # 3. Invalid values by field (bar chart).
    invalid = pd.Series({
        "invalid_email": int(df["invalid_email"].sum()),
        "invalid_phone": int(df["invalid_phone"].sum()),
        "invalid_url": int(df["invalid_url"].sum()),
        "token_label_mismatch": int(df["token_label_mismatch"].sum()),
        "invalid_labels": int(df["invalid_labels"].sum()),
    }).sort_values(ascending=False)
    if not invalid.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        invalid.plot(kind="bar", ax=ax, color="firebrick")
        ax.set_title("Invalid / Malformed Values by Field")
        ax.set_xlabel("Validation flag")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, viz_dir / "invalid_values.png")

    # 4. Duplicate / repeated-data counts (bar chart).
    dup_counts = pd.Series({
        "duplicate_email": int(df["duplicate_email"].sum()),
        "duplicate_phone": int(df["duplicate_phone"].sum()),
        "duplicate_identity": int(df["duplicate_identity"].sum()),
    }).fillna(0)
    if not dup_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        dup_counts.plot(kind="bar", ax=ax, color="mediumpurple")
        ax.set_title("Duplicate / Repeated Contact Data")
        ax.set_xlabel("Duplicate type")
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, viz_dir / "duplicate_values.png")

    # 5. Top jobs (horizontal bar).
    _plot_top_categorical(df, "job", "Top Jobs", viz_dir / "top_jobs.png")

    # 6. Top hobbies (horizontal bar).
    _plot_top_categorical(df, "hobby", "Top Hobbies", viz_dir / "top_hobbies.png")

    # 7. Text length histogram.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["text_length"], bins=30, kde=True, ax=ax, color="steelblue")
    ax.set_title("Document Text Length Distribution")
    ax.set_xlabel("Text length (characters)")
    _save_fig(fig, viz_dir / "text_length_histogram.png")

    # 8. Text length boxplot.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["text_length"], ax=ax, color="lightblue")
    ax.set_title("Document Text Length Boxplot (outlier view)")
    ax.set_xlabel("Text length (characters)")
    _save_fig(fig, viz_dir / "text_length_boxplot.png")

    # 9. PII count histogram.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["pii_entity_count"], bins=15, kde=True, ax=ax, color="seagreen")
    ax.set_title("PII Entity Count Distribution")
    ax.set_xlabel("Number of PII entities")
    _save_fig(fig, viz_dir / "pii_count_histogram.png")

    # 10. PII count boxplot.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["pii_entity_count"], ax=ax, color="lightgreen")
    ax.set_title("PII Entity Count Boxplot")
    ax.set_xlabel("Number of PII entities")
    _save_fig(fig, viz_dir / "pii_count_boxplot.png")

    # 11. Text length vs PII entity count (scatter).
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        df["text_length"], df["pii_entity_count"],
        c=df["anomaly_flag"], cmap="coolwarm", alpha=0.6, s=20,
    )
    ax.set_title("Text Length vs PII Entity Count")
    ax.set_xlabel("Text length (characters)")
    ax.set_ylabel("PII entity count")
    fig.colorbar(sc, ax=ax, label="Anomaly flag")
    _save_fig(fig, viz_dir / "text_length_vs_pii_count.png")

    # 12. Correlation heatmap of meaningful derived features.
    corr_cols = ["text_length", "token_count", "pii_token_count",
                 "pii_entity_count", "pii_ratio", "unique_token_count"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap: Derived Features")
        _save_fig(fig, viz_dir / "correlation_heatmap.png")

    # 13. Quality categories (donut chart).
    qc = df["quality_category"].value_counts()
    if not qc.empty:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            qc.values, labels=qc.index, autopct="%1.1f%%",
            startangle=90, colors=sns.color_palette("Set2", len(qc)),
            wedgeprops={"width": 0.4},
        )
        ax.set_title("Data Quality Category Distribution")
        _save_fig(fig, viz_dir / "quality_categories.png")

    # 14. Before vs after quality metrics (bar chart).
    if before_after:
        pairs = [
            ("invalid_email", before_after.get("raw_invalid_email", 0),
             before_after.get("processed_invalid_email", 0)),
            ("invalid_phone", before_after.get("raw_invalid_phone", 0),
             before_after.get("processed_invalid_phone", 0)),
            ("invalid_url", before_after.get("raw_invalid_url", 0),
             before_after.get("processed_invalid_url", 0)),
        ]
        labels = [p[0] for p in pairs]
        raw_vals = [p[1] for p in pairs]
        proc_vals = [p[2] for p in pairs]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 6))
        b1 = ax.bar(x - width / 2, raw_vals, width, label="Raw", color="gray")
        b2 = ax.bar(x + width / 2, proc_vals, width, label="Processed", color="teal")
        ax.set_title("Before vs After: Invalid Values")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count")
        ax.legend()
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{int(h)}", (b.get_x() + b.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)
        _save_fig(fig, viz_dir / "before_after_quality.png")

    logger.info("Visualizations written to %s", viz_dir)


def _plot_top_categorical(
    df: pd.DataFrame, col: str, title: str, path: Path, top: int = 12
) -> None:
    if col not in df.columns:
        return
    counts = df[col].dropna().value_counts().head(top)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    counts.iloc[::-1].plot(kind="barh", ax=ax, color="mediumseagreen")
    ax.set_title(title)
    ax.set_xlabel("Count")
    _save_fig(fig, path)


# ============================================================
# SQLITE DATABASE
# ============================================================


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    document_id          TEXT PRIMARY KEY,
    text                 TEXT,
    text_length          INTEGER,
    prompt               TEXT,
    prompt_id            INTEGER,
    job                  TEXT,
    hobby                TEXT
);

CREATE TABLE IF NOT EXISTS pii_entities (
    entity_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id          TEXT NOT NULL,
    entity_type          TEXT NOT NULL,
    entity_value         TEXT,
    token_start          INTEGER,
    token_end            INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS contact_information (
    document_id          TEXT PRIMARY KEY,
    name                 TEXT,
    email                TEXT,
    email_normalized     TEXT,
    email_valid          INTEGER,
    phone                TEXT,
    phone_normalized     TEXT,
    phone_valid          INTEGER,
    address              TEXT,
    username             TEXT,
    url                  TEXT,
    url_normalized       TEXT,
    url_valid            INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS data_quality (
    document_id          TEXT PRIMARY KEY,
    missing_count        INTEGER,
    duplicate_flag       INTEGER,
    invalid_email        INTEGER,
    invalid_phone        INTEGER,
    invalid_url          INTEGER,
    token_label_mismatch INTEGER,
    anomaly_flag         INTEGER,
    anomaly_score        REAL,
    quality_score        REAL,
    quality_category     TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id          TEXT NOT NULL,
    anomaly_type         TEXT,
    description          TEXT,
    severity             REAL,
    anomaly_score        REAL,
    source               TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS etl_runs (
    run_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp        TEXT,
    input_rows           INTEGER,
    processed_rows       INTEGER,
    flagged_rows         INTEGER,
    anomalies_detected   INTEGER,
    status               TEXT
);

CREATE INDEX IF NOT EXISTS idx_entities_type   ON pii_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_contact_email    ON contact_information(email_normalized);
CREATE INDEX IF NOT EXISTS idx_contact_phone    ON contact_information(phone_normalized);
CREATE INDEX IF NOT EXISTS idx_quality_anomaly  ON data_quality(anomaly_flag);
CREATE INDEX IF NOT EXISTS idx_quality_category ON data_quality(quality_category);
"""


def _extract_entities(row: pd.Series) -> List[Tuple[str, str, int, int]]:
    """Extract (entity_type, entity_value, token_start, token_end) spans."""
    tokens = row.get("tokens_parsed")
    labels = row.get("labels_parsed")
    if not isinstance(tokens, list) or not isinstance(labels, list):
        return []
    if len(tokens) != len(labels):
        return []
    entities: List[Tuple[str, str, int, int]] = []
    start: Optional[int] = None
    cur_type: Optional[str] = None
    for i, (tok, lab) in enumerate(zip(tokens, labels)):
        if not isinstance(lab, str) or lab == "O":
            if start is not None and cur_type:
                val = " ".join(str(x) for x in tokens[start:i])
                entities.append((cur_type, val, start, i - 1))
            start, cur_type = None, None
            continue
        prefix = lab.split("-", 1)[0] if "-" in lab else ""
        etype = lab.split("-", 1)[1] if "-" in lab else lab
        if prefix == "B":
            if start is not None and cur_type:
                val = " ".join(str(x) for x in tokens[start:i])
                entities.append((cur_type, val, start, i - 1))
            start, cur_type = i, etype
        elif prefix == "I":
            if start is None:
                # I-token without B start; treat as begin of its own span.
                start, cur_type = i, etype
            else:
                cur_type = etype
        else:
            if start is not None and cur_type:
                val = " ".join(str(x) for x in tokens[start:i])
                entities.append((cur_type, val, start, i - 1))
            start, cur_type = None, None
    if start is not None and cur_type:
        val = " ".join(str(x) for x in tokens[start:])
        entities.append((cur_type, val, start, len(tokens) - 1))
    return entities


def create_sqlite_database(db_path: Path, logger: logging.Logger) -> sqlite3.Connection:
    """Create the SQLite database with the full schema. Returns the connection."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("SQLite database created: %s", db_path)
    except sqlite3.Error as exc:
        conn.close()
        raise RuntimeError(f"Failed to create SQLite schema: {exc}") from exc
    return conn


def load_data_into_sqlite(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    anomaly_records: List[Dict[str, Any]],
    input_rows: int,
    flagged_rows: int,
    logger: logging.Logger,
) -> int:
    """Load the cleaned DataFrame + anomalies + run metadata into SQLite.

    Uses parameterized inserts wrapped in a single transaction. Returns the
    number of document rows loaded.
    """
    try:
        with conn:
            cursor = conn.cursor()

            # documents
            doc_rows = [
                (
                    row["document"], row["text"], int(row["text_length"]),
                    row.get("prompt"), _int_or_none(row.get("prompt_id")),
                    row.get("job"), row.get("hobby"),
                )
                for _, row in df.iterrows()
            ]
            cursor.executemany(
                "INSERT OR REPLACE INTO documents "
                "(document_id, text, text_length, prompt, prompt_id, job, hobby) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                doc_rows,
            )

            # contact_information
            contact_rows = []
            for _, row in df.iterrows():
                contact_rows.append((
                    row["document"],
                    row.get("name"),
                    row.get("email"),
                    row.get("email_normalized"),
                    _bool_int(row.get("email_valid")),
                    row.get("phone"),
                    row.get("phone_normalized"),
                    _bool_int(row.get("phone_valid")),
                    row.get("address"),
                    row.get("username"),
                    row.get("url"),
                    row.get("url_normalized"),
                    _bool_int(row.get("url_valid")),
                ))
            cursor.executemany(
                "INSERT OR REPLACE INTO contact_information "
                "(document_id, name, email, email_normalized, email_valid, "
                " phone, phone_normalized, phone_valid, address, username, "
                " url, url_normalized, url_valid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                contact_rows,
            )

            # data_quality
            quality_rows = []
            for _, row in df.iterrows():
                quality_rows.append((
                    row["document"],
                    int(row.get("missing_count", 0) or 0),
                    _bool_int(row.get("duplicate_flag")),
                    _bool_int(row.get("invalid_email")),
                    _bool_int(row.get("invalid_phone")),
                    _bool_int(row.get("invalid_url")),
                    _bool_int(row.get("token_label_mismatch")),
                    _bool_int(row.get("anomaly_flag")),
                    _float_or_none(row.get("anomaly_score")),
                    _float_or_none(row.get("quality_score")),
                    str(row.get("quality_category", "")),
                ))
            cursor.executemany(
                "INSERT OR REPLACE INTO data_quality "
                "(document_id, missing_count, duplicate_flag, invalid_email, "
                " invalid_phone, invalid_url, token_label_mismatch, anomaly_flag, "
                " anomaly_score, quality_score, quality_category) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                quality_rows,
            )

            # pii_entities
            entity_rows = []
            for _, row in df.iterrows():
                for etype, evalue, ts, te in _extract_entities(row):
                    # Never store raw sensitive entity values in a way that's
                    # printed; storage in DB for analytics is intended though.
                    entity_rows.append((
                        row["document"], etype, evalue[:200], ts, te,
                    ))
            cursor.executemany(
                "INSERT OR REPLACE INTO pii_entities "
                "(document_id, entity_type, entity_value, token_start, token_end) "
                "VALUES (?, ?, ?, ?, ?)",
                entity_rows,
            )

            # anomalies
            cursor.executemany(
                "INSERT OR REPLACE INTO anomalies "
                "(document_id, anomaly_type, description, severity, anomaly_score, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        rec["document_id"], rec["anomaly_type"],
                        rec["description"], rec["severity"],
                        rec["anomaly_score"], rec["source"],
                    )
                    for rec in anomaly_records
                ],
            )

            # etl_runs
            cursor.execute(
                "INSERT INTO etl_runs "
                "(run_timestamp, input_rows, processed_rows, flagged_rows, "
                " anomalies_detected, status) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    int(input_rows),
                    int(len(df)),
                    int(flagged_rows),
                    int(len(anomaly_records)),
                    "success",
                ),
            )

            conn.commit()
        logger.info(
            "Loaded %d documents, %d contact rows, %d quality rows, "
            "%d entities and %d anomalies into SQLite.",
            len(df), len(df), len(df), len(entity_rows), len(anomaly_records),
        )
        return len(df)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to load data into SQLite: {exc}") from exc


def _bool_int(v: Any) -> int:
    try:
        return int(bool(v))
    except (TypeError, ValueError):
        return 0


def _int_or_none(v: Any) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ============================================================
# SQL ANALYTICS
# ============================================================


def run_sql_queries(conn: sqlite3.Connection, logger: logging.Logger) -> dict:
    """Execute a set of meaningful SQL analytics and print results."""
    analytics: Dict[str, Any] = {}

    def q(title: str, sql: str) -> None:
        try:
            rows = conn.execute(sql).fetchall()
            analytics[title] = rows
            print(f"\n--- {title} ---")
            if not rows:
                print("  (no rows)")
                return
            # Print header (first column names) plus up to 15 rows.
            cols = [d[0] for d in conn.execute(sql).description]
            widths = [max(len(str(c)), *(len(str(r[i] or "")) for r in rows[:15] if r)) for i, c in enumerate(cols)]
            header = "  " + "  |  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))
            print(header)
            print("  " + "-" * len(header))
            for r in rows[:15]:
                print("  " + "  |  ".join(str((r[i] if r[i] is not None else "")).ljust(widths[i]) for i in range(len(cols))))
            if len(rows) > 15:
                print(f"  ... and {len(rows) - 15} more rows")
        except sqlite3.Error as exc:
            logger.error("SQL analytics failed for %r: %s", title, exc)

    q("1. Total number of documents",
      "SELECT COUNT(*) AS total_documents FROM documents;")

    q("2. PII entity distribution",
      "SELECT entity_type, COUNT(*) AS count FROM pii_entities "
      "GROUP BY entity_type ORDER BY count DESC;")

    q("3. Missing values per contact field",
      "SELECT "
      " SUM(CASE WHEN name IS NULL OR name='' THEN 1 ELSE 0 END) AS missing_name,"
      " SUM(CASE WHEN email IS NULL OR email='' THEN 1 ELSE 0 END) AS missing_email,"
      " SUM(CASE WHEN phone IS NULL OR phone='' THEN 1 ELSE 0 END) AS missing_phone,"
      " SUM(CASE WHEN username IS NULL OR username='' THEN 1 ELSE 0 END) AS missing_username,"
      " SUM(CASE WHEN url IS NULL OR url='' THEN 1 ELSE 0 END) AS missing_url"
      " FROM contact_information;")

    q("4. Invalid email count",
      "SELECT COUNT(*) AS invalid_emails FROM contact_information "
      "WHERE email_valid = 0 AND (email IS NOT NULL AND email <> '');")

    q("5. Invalid phone count",
      "SELECT COUNT(*) AS invalid_phones FROM contact_information "
      "WHERE phone_valid = 0 AND (phone IS NOT NULL AND phone <> '');")

    q("6. Invalid URL count",
      "SELECT COUNT(*) AS invalid_urls FROM contact_information "
      "WHERE url_valid = 0 AND (url IS NOT NULL AND url <> '');")

    q("7. Duplicate identity records",
      "SELECT COUNT(*) AS duplicate_identity FROM data_quality "
      "WHERE duplicate_flag = 1;")

    q("8. Quality-category distribution",
      "SELECT quality_category, COUNT(*) AS count FROM data_quality "
      "GROUP BY quality_category ORDER BY count DESC;")

    q("9. Anomaly distribution by type",
      "SELECT anomaly_type, COUNT(*) AS count, ROUND(AVG(severity),2) AS avg_severity "
      "FROM anomalies GROUP BY anomaly_type ORDER BY count DESC;")

    q("10. Top jobs",
      "SELECT job, COUNT(*) AS count FROM documents "
      "WHERE job IS NOT NULL GROUP BY job ORDER BY count DESC LIMIT 15;")

    q("11. Top hobbies",
      "SELECT hobby, COUNT(*) AS count FROM documents "
      "WHERE hobby IS NOT NULL GROUP BY hobby ORDER BY count DESC LIMIT 15;")

    q("12. Documents with highest PII density",
      "SELECT d.document_id, d.text_length, "
      " (SELECT COUNT(*) FROM pii_entities e WHERE e.document_id = d.document_id) AS pii_count, "
      " ROUND(CAST((SELECT COUNT(*) FROM pii_entities e "
      "   WHERE e.document_id = d.document_id) AS REAL) / NULLIF(d.text_length,0), 4) AS pii_density "
      "FROM documents d ORDER BY pii_density DESC LIMIT 10;")

    q("13. Documents with highest anomaly scores",
      "SELECT document_id, ROUND(anomaly_score,4) AS anomaly_score, quality_category "
      "FROM data_quality ORDER BY anomaly_score DESC LIMIT 10;")

    q("14. Duplicate emails (appearing more than once)",
      "SELECT email_normalized, COUNT(*) AS count FROM contact_information "
      "WHERE email_normalized IS NOT NULL AND email_normalized <> '' "
      "GROUP BY email_normalized HAVING COUNT(*) > 1 ORDER BY count DESC;")

    q("15. Duplicate phone numbers",
      "SELECT phone_normalized, COUNT(*) AS count FROM contact_information "
      "WHERE phone_normalized IS NOT NULL AND phone_normalized <> '' "
      "GROUP BY phone_normalized HAVING COUNT(*) > 1 ORDER BY count DESC;")

    q("16. Average document length",
      "SELECT ROUND(AVG(text_length),2) AS avg_text_length FROM documents;")

    q("17. Average PII entity count per document",
      "SELECT ROUND(AVG(ec),2) AS avg_pii_entities "
      "FROM (SELECT document_id, COUNT(*) AS ec FROM pii_entities "
      "      GROUP BY document_id);")

    q("18. ETL run statistics",
      "SELECT run_id, run_timestamp, input_rows, processed_rows, "
      " flagged_rows, anomalies_detected, status FROM etl_runs "
      "ORDER BY run_id DESC LIMIT 5;")

    return analytics


# ============================================================
# BEFORE / AFTER ANALYSIS
# ============================================================


def build_before_after(
    raw_df: pd.DataFrame, processed_df: pd.DataFrame, profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Produce before vs after data-quality metrics for reporting."""
    raw_missing = _detect_missing_mask(
        raw_df[["name", "email", "phone", "job", "address", "username", "url", "hobby"]]
    )
    proc_missing = _detect_missing_mask(
        processed_df[["name", "email", "phone", "job", "address", "username", "url", "hobby"]]
    )
    raw_invalid_email = int(
        (~raw_df["email"].map(lambda x: bool(EMAIL_RE.fullmatch(str(x).strip().lower()))
                              if isinstance(x, str) and str(x).strip() else False)).sum()
    )
    proc_invalid_email = int(processed_df["invalid_email"].sum())
    raw_invalid_phone = int(_count_invalid_phone_raw(raw_df["phone"]))
    proc_invalid_phone = int(processed_df["invalid_phone"].sum())
    raw_invalid_url = int(_count_invalid_url_raw(raw_df["url"]))
    proc_invalid_url = int(processed_df["invalid_url"].sum())

    return {
        "raw_rows": int(profile["rows"]),
        "processed_rows": int(len(processed_df)),
        "raw_missing_total": int(raw_missing.sum().sum()),
        "processed_missing_total": int(proc_missing.sum().sum()),
        "raw_invalid_email": raw_invalid_email,
        "processed_invalid_email": proc_invalid_email,
        "raw_invalid_phone": raw_invalid_phone,
        "processed_invalid_phone": proc_invalid_phone,
        "raw_invalid_url": raw_invalid_url,
        "processed_invalid_url": proc_invalid_url,
        "duplicate_issues": int(
            processed_df["duplicate_email"].sum() + processed_df["duplicate_phone"].sum()
        ),
        "token_label_mismatches": int(processed_df["token_label_mismatch"].sum()),
        "anomalies": int((processed_df["anomaly_any"] == 1).sum()),
        "reviews_needed": int((processed_df["quality_category"] != "Excellent").sum()),
        "avg_quality": float(processed_df["quality_score"].mean()),
    }


def _count_invalid_phone_raw(series: pd.Series) -> int:
    cnt = 0
    for v in series:
        if pd.isna(v) or not isinstance(v, str) or not v.strip():
            continue
        if not PHONE_RE.fullmatch(v.strip()):
            cnt += 1
            continue
        digits = re.sub(r"\D", "", v)
        if not (6 <= len(digits) <= 15):
            cnt += 1
    return cnt


def _count_invalid_url_raw(series: pd.Series) -> int:
    cnt = 0
    for v in series:
        if pd.isna(v) or not isinstance(v, str) or not v.strip():
            continue
        if not URL_RE.fullmatch(v.strip()):
            cnt += 1
    return cnt


# ============================================================
# FINAL VALIDATION
# ============================================================


def final_validation(
    raw_csv_path: Path,
    db_path: Path,
    conn: sqlite3.Connection,
    expected_rows: int,
    viz_dir: Path,
    logger: logging.Logger,
) -> None:
    """Verify the pipeline outputs before declaring success."""
    checks: List[Tuple[str, bool]] = []

    checks.append(("Original CSV still exists", raw_csv_path.exists()))

    # Compare raw file hash before/after is ideal; we track that externally.
    checks.append(("SQLite database exists", db_path.exists()))

    # Required tables exist.
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    required_tables = {"documents", "pii_entities", "contact_information",
                       "data_quality", "anomalies", "etl_runs"}
    checks.append(("All required tables present", required_tables.issubset(tables)))

    # Processed record count.
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    checks.append((f"Processed rows in documents ({doc_count}) == {expected_rows}",
                   doc_count == expected_rows))

    # Foreign-key integrity: no orphaned contact rows.
    orphans = conn.execute(
        "SELECT COUNT(*) FROM contact_information ci "
        "LEFT JOIN documents d ON ci.document_id = d.document_id "
        "WHERE d.document_id IS NULL"
    ).fetchone()[0]
    checks.append(("No orphan foreign keys in contact_information", orphans == 0))

    # Quality scores present.
    q_score_null = conn.execute(
        "SELECT COUNT(*) FROM data_quality WHERE quality_score IS NULL"
    ).fetchone()[0]
    checks.append(("Quality scores populated", q_score_null == 0))

    # Anomalies exist where flagged.
    flagged = conn.execute(
        "SELECT COUNT(*) FROM data_quality WHERE anomaly_flag = 1"
    ).fetchone()[0]
    anomalies = conn.execute("SELECT COUNT(*) FROM anomalies WHERE source='ml'").fetchone()[0]
    checks.append(("ML anomalies stored (flagged=%d, ml_rows=%d)" % (flagged, anomalies),
                   anomalies >= flagged))

    # SQL query executes.
    try:
        conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        checks.append(("SQL queries execute successfully", True))
    except sqlite3.Error:
        checks.append(("SQL queries execute successfully", False))

    # Visualizations generated.
    pngs = list(viz_dir.glob("*.png")) if viz_dir.exists() else []
    checks.append(("Visualizations generated (%d PNGs)" % len(pngs), len(pngs) > 0))

    # Report results.
    print("\n==================== FINAL VALIDATION ====================")
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}")
    print("============================================================")
    if all_ok:
        logger.info("Final validation: ALL CHECKS PASSED")
    else:
        logger.error("Final validation: SOME CHECKS FAILED")
    if not all_ok:
        raise RuntimeError("Final validation failed. See validation summary above.")


# ============================================================
# PIPELINE SUMMARY / REPORT
# ============================================================


def print_pipeline_summary(cfg: Dict[str, Any], ba: Dict[str, Any]) -> None:
    """Print a concise, PII-safe end-of-run summary."""
    separator = "=" * 46
    print(f"\n{separator}")
    print("         ETL PIPELINE COMPLETED")
    print(separator)
    print()
    print(f"Input records:        {ba['raw_rows']}")
    print(f"Processed records:    {ba['processed_rows']}")
    print(f"Missing values:       {ba['processed_missing_total']}")
    print(f"Invalid emails:       {ba['processed_invalid_email']}")
    print(f"Invalid phones:       {ba['processed_invalid_phone']}")
    print(f"Invalid URLs:         {ba['processed_invalid_url']}")
    print(f"Duplicate issues:     {ba['duplicate_issues']}")
    print(f"Token/label mismatch: {ba['token_label_mismatches']}")
    print(f"Anomalies:            {ba['anomalies']}")
    print(f"Records needing review: {ba['reviews_needed']}")
    print(f"Average quality score:  {ba['avg_quality']:.2f}")
    print()
    print(f"SQLite database:      {cfg['database_file']}")
    print("Visualizations:       visualizations/")
    print()
    print(separator)


# ============================================================
# MAIN PIPELINE
# ============================================================


def main() -> None:
    """Orchestrate the full ETL -> quality -> analytics pipeline."""
    cfg = load_config()
    logger = setup_logging(cfg["log_file"])

    logger.info("Starting ETL pipeline")
    logger.info("Loading configuration")
    logger.info("Configuration: input=%s db=%s", cfg["input_file"], cfg["database_file"])

    project_dir = Path(__file__).resolve().parent
    input_path = project_dir / cfg["input_file"]
    db_path = project_dir / cfg["database_file"]
    viz_dir = project_dir / cfg["visualizations_dir"]
    log_path = project_dir / cfg["log_file"]

    # Record a hash of the raw file up front to prove it is not modified later.
    from hashlib import sha256

    def _file_hash(p: Path) -> str:
        h = sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    raw_hash_before = _file_hash(input_path) if input_path.exists() else None

    # -------------------- EXTRACT --------------------
    logger.info("Loading dataset")
    df = load_dataset(cfg["input_file"])
    logger.info("Dataset loaded: %d rows, %d columns", df.shape[0], df.shape[1])

    # -------------------- PROFILE --------------------
    logger.info("Profiling dataset")
    raw_missing = _detect_missing_mask(df)
    profile = profile_dataset(df, raw_missing, logger)

    # -------------------- CLEAN --------------------
    logger.info("Cleaning data")
    df_clean, raw_total, cleaned_total = clean_data(df, logger)

    # -------------------- TOKENS / LABELS --------------------
    logger.info("Parsing tokens and labels")
    df_clean = parse_tokens_and_labels(df_clean, logger)
    df_clean = add_derived_features(df_clean)

    # -------------------- VALIDATION FLAGS --------------------
    logger.info("Validating data")
    df_clean = _flag_annotations(df_clean)

    # -------------------- DUPLICATES --------------------
    logger.info("Detecting duplicates")
    df_clean = detect_duplicates(df_clean)
    repeated = analyze_repeated_values(df_clean)
    logger.info("Repeated-value analysis done: %s", repeated)

    # -------------------- ANOMALIES --------------------
    logger.info("Detecting anomalies")
    df_clean, anomaly_records = detect_anomalies(
        df_clean, cfg["anomaly_contamination"], cfg["random_seed"], logger
    )

    # -------------------- QUALITY SCORING --------------------
    logger.info("Calculating quality scores")
    df_clean = calculate_quality_scores(df_clean)

    # -------------------- BEFORE / AFTER --------------------
    ba = build_before_after(df, df_clean, profile)
    logger.info(
        "Before/after: raw missing=%d processed missing=%d avg quality=%.2f",
        ba["raw_missing_total"], ba["processed_missing_total"], ba["avg_quality"],
    )

    # -------------------- VISUALIZATIONS --------------------
    logger.info("Generating visualizations")
    try:
        generate_visualizations(df_clean, ba, viz_dir, logger)
    except Exception as exc:  # noqa: BLE001 - visualizations are non-critical
        logger.error("Visualization generation failed: %s", exc)
        raise

    # -------------------- SQLITE --------------------
    logger.info("Creating SQLite database")
    conn = create_sqlite_database(db_path, logger)
    try:
        flagged_rows = int((df_clean["anomaly_any"] == 1).sum())
        loaded = load_data_into_sqlite(
            conn, df_clean, anomaly_records,
            input_rows=profile["rows"], flagged_rows=flagged_rows, logger=logger,
        )
        logger.info("Loading SQLite database complete: %d documents", loaded)

        # -------------------- SQL ANALYTICS --------------------
        logger.info("Running analytics")
        run_sql_queries(conn, logger)

        # -------------------- FINAL VALIDATION --------------------
        logger.info("Final validation")
        raw_hash_after = _file_hash(input_path) if input_path.exists() else None
        final_validation(input_path, db_path, conn, profile["rows"], viz_dir, logger)

        if raw_hash_before is not None and raw_hash_before != raw_hash_after:
            raise RuntimeError("The raw input CSV was modified during the pipeline!")

    finally:
        conn.close()

    # -------------------- SUMMARY --------------------
    print_pipeline_summary(cfg, ba)
    logger.info("ETL pipeline completed successfully")

    # Remove the log file from project dir by default config? No - keep it.
    if not log_path.exists():
        # Recreate via earlier handler; but it was created at setup. Keep as is.
        pass


if __name__ == "__main__":
    main()
