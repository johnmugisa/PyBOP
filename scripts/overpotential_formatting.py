# overpotential_formatting.py
import json
import pandas as pd

class Overpotential_Information:
    """
    Minimal Measurement-like wrapper for overpotential data, compatible with
    ep_bolfi.utility.dataset_formatting.store_parquet_table().
    Provides:
      - from_json(text) -> Overpotential_Information
      - to_dataframe() -> pd.DataFrame
      - example_table_row() -> dict
      - table_descriptors() -> list[dict]
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @classmethod
    def from_json(cls, text: str):
        data = json.loads(text)
        # Required keys (as seen in your file)
        tp = data["timepoints [s]"]
        ia = data["currents [A]"]
        vv = data["voltages [V]"]
        idx = data.get("indices", list(range(len(tp))))

        n = len(tp)
        # "other_columns" is optional and may be None or missing
        other = data.get("other_columns", None)
        if other is None or isinstance(other, (str, int, float, dict)):
            other = [None] * n
        else:
            # If present but shorter, pad with None
            if len(other) < n:
                other = list(other) + [None] * (n - len(other))

        rows = []
        for i in range(n):
            t_list, c_list, v_list = tp[i], ia[i], vv[i]
            pulse = idx[i]
            extra = other[i] if i < len(other) else None
            # zip to avoid length mismatches
            for t, c, v in zip(t_list, c_list, v_list):
                rows.append({
                    "time [s]": t,
                    "current [A]": c,
                    "voltage [V]": v,
                    "pulse_index": pulse,
                    "other": extra,
                })

        df = pd.DataFrame(rows)
        # enforce dtypes where sensible
        if not df.empty:
            df["pulse_index"] = df["pulse_index"].astype("int32", errors="ignore")
        return cls(df)

    def to_dataframe(self) -> pd.DataFrame:
        return self._df

    def example_table_row(self) -> dict:
        if self._df.empty:
            return {
                "time [s]": 0.0,
                "current [A]": 0.0,
                "voltage [V]": 0.0,
                "pulse_index": 0,
                "other": None,
            }
        return self._df.iloc[0].to_dict()

    def table_descriptors(self):
        # EP-BOLFI expects a list of descriptor dicts with at least name/unit/dtype
        # (mimicking Cycling_Information style)
        return [
            {"name": "time [s]",     "unit": "s", "dtype": "float64"},
            {"name": "current [A]",  "unit": "A", "dtype": "float64"},
            {"name": "voltage [V]",  "unit": "V", "dtype": "float64"},
            {"name": "pulse_index",  "unit": "-", "dtype": "int32"},
            {"name": "other",        "unit": "-", "dtype": "object"},
        ]
