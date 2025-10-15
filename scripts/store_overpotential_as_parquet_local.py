# store_overpotential_as_parquet_local.py
"""
Write overpotential JSON (via stdin) into a Parquet file where each segment
is a separate ROW GROUP, while keeping a flat, pandas-friendly schema.

Columns:
  - time [s]      (float64)
  - current [A]   (float64)
  - voltage [V]   (float64)
  - pulse_index   (int32 or string; repeated per row within a segment)
  - other         (string, currently None)

Usage (PowerShell):
  Get-Content C:\...\overpotentials.json |
    python C:\...\store_overpotential_as_parquet_local.py `
      -n C:\...\overpotentials `
      -c 22 `
      --overwrite
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def _as_list(x, length: int):
    """Return x as list of len=length. Scalar -> broadcast."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x] * length


def _flatten_segment(obj: dict, seg_idx: int):
    """Return arrays (time, current, voltage) and a pulse_index for a segment."""
    times_segments = obj["timepoints [s]"]
    currents_segments = obj["currents [A]"]
    voltages_segments = obj["voltages [V]"]
    indices = obj.get("indices")

    ts = times_segments[seg_idx]
    if not isinstance(ts, (list, tuple, np.ndarray)):
        raise TypeError(f"timepoints segment {seg_idx} is not a list/array")

    cs = _as_list(currents_segments[seg_idx], len(ts))
    vs = _as_list(voltages_segments[seg_idx], len(ts))
    if len(cs) != len(ts) or len(vs) != len(ts):
        raise ValueError(
            f"Length mismatch in segment {seg_idx}: "
            f"len(times)={len(ts)} len(currents)={len(cs)} len(voltages)={len(vs)}"
        )

    # Resolve pulse_index for this segment
    if indices and len(indices) == len(times_segments):
        pidx_raw = indices[seg_idx]
    else:
        pidx_raw = seg_idx  # fallback

    return np.asarray(ts, dtype="float64"), np.asarray(cs, dtype="float64"), np.asarray(vs, dtype="float64"), pidx_raw


@click.command()
@click.option(
    "-n", "--filename", required=True, type=str,
    help="Output Parquet base name; '.parquet' is appended if missing."
)
@click.option(
    "-c", "--compression-level", default=22, show_default=True, type=int,
    help="Zstandard compression level (-7 fastest/largest, 22 slowest/smallest)."
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing file.")
def main(filename: str, compression_level: int, overwrite: bool):
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("No JSON received on stdin. Pipe your overpotentials.json into this script.")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse JSON from stdin: {e}")

    # Validate required keys
    for k in ["timepoints [s]", "currents [A]", "voltages [V]"]:
        if k not in obj:
            raise KeyError(f"Missing required key '{k}' in JSON. Keys present: {list(obj.keys())}")

    n_segments = len(obj["timepoints [s]"])
    if not (len(obj["currents [A]"]) == len(obj["voltages [V]"]) == n_segments):
        raise ValueError("timepoints/currents/voltages lists must have equal length (per segments).")

    # Decide pulse_index dtype (prefer int if possible)
    indices = obj.get("indices", list(range(n_segments)))
    use_int_pidx = True
    try:
        _ = [int(i) for i in indices]
    except Exception:
        use_int_pidx = False

    # Build schema with metadata
    fields = [
        pa.field("time [s]", pa.float64()),
        pa.field("current [A]", pa.float64()),
        pa.field("voltage [V]", pa.float64()),
        pa.field("pulse_index", pa.int32() if use_int_pidx else pa.string()),
        pa.field("other", pa.string()),  # keep a placeholder column
    ]
    # File-level metadata: ordered segment IDs
    kv_meta = {
        "writer": "overpotential_flat_rowgroups_v1",
        "segment_ids_json": json.dumps([int(i) if use_int_pidx else str(i) for i in indices]),
    }
    schema = pa.schema(fields, metadata={k: v.encode("utf-8") for k, v in kv_meta.items()})

    out_path = Path(filename)
    if out_path.suffix.lower() != ".parquet":
        out_path = out_path.with_suffix(".parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to replace it.")

    # Write one row group per segment
    writer = pq.ParquetWriter(
        where=str(out_path),
        schema=schema,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=True,
    )

    total_rows = 0
    for seg_idx in range(n_segments):
        t, c, v, pidx_raw = _flatten_segment(obj, seg_idx)

        if use_int_pidx:
            pidx_arr = pa.array(np.full(t.shape[0], int(pidx_raw), dtype="int32"))
        else:
            pidx_arr = pa.array([str(pidx_raw)] * t.shape[0], type=pa.string())

        table = pa.Table.from_arrays(
            [
                pa.array(t, type=pa.float64()),
                pa.array(c, type=pa.float64()),
                pa.array(v, type=pa.float64()),
                pidx_arr,
                pa.array([None] * t.shape[0], type=pa.string()),
            ],
            names=[f.name for f in fields],
        )
        writer.write_table(table)
        total_rows += table.num_rows

    writer.close()

    # Quick verification using pandas (works with row-groups too)
    df_check = pd.read_parquet(out_path)
    print(f"shape: {df_check.shape}")
    print(f"columns: {list(df_check.columns)}")
    with pd.option_context("display.max_rows", 5, "display.width", 120):
        if not df_check.empty:
            print(df_check.head(3).to_string(index=False))
    print(f"Wrote: {str(out_path)}")


if __name__ == "__main__":
    main()
