from __future__ import annotations

import pandas as pd

from src.data_loader import load_all_data


EXPECTED_DATASETS = {
    "employees",
    "budgets",
    "software_catalog",
    "vendors",
    "purchase_history",
    "requests",
    "vendor_risk",
    "policies",
}


def test_load_all_data_returns_non_empty_dataframes() -> None:
    data = load_all_data()

    assert set(data) == EXPECTED_DATASETS
    assert all(isinstance(frame, pd.DataFrame) for frame in data.values())
    assert all(not frame.empty for frame in data.values())
