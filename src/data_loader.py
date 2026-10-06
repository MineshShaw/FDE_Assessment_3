from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _load_json_records(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        return pd.DataFrame.from_dict(payload, orient="index").reset_index(names="vendor_name")
    raise ValueError(f"Unsupported JSON structure in {path.name}")


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load the complete synthetic data snapshot into Pandas DataFrames."""
    policy_path = DATA_DIR / "procurement_policy.md"
    return {
        "employees": pd.read_csv(DATA_DIR / "employees.csv"),
        "budgets": pd.read_csv(DATA_DIR / "department_budgets.csv"),
        "software_catalog": pd.read_csv(DATA_DIR / "software_catalog.csv"),
        "vendors": pd.read_csv(DATA_DIR / "vendors.csv"),
        "purchase_history": pd.read_csv(DATA_DIR / "purchase_history.csv"),
        "requests": _load_json_records(DATA_DIR / "requests.json"),
        "vendor_risk": _load_json_records(DATA_DIR / "vendor_risk.json"),
        "policies": pd.DataFrame(
            [{"policy_name": policy_path.stem, "content": policy_path.read_text(encoding="utf-8")}]
        ),
    }
