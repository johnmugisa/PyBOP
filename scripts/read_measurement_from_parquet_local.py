"""
LOCAL-ONLY replacement for:
  python -m ep_bolfi.kadi_tools.read_measurement_from_parquet

Reads a Measurement object from a local Apache Parquet file (as produced by
ep_bolfi.kadi_tools.store_measurement_in_parquet) and prints EP-BOLFI JSON
to stdout. No Kadi access.

Usage (same flags; -r/--record is accepted but ignored):
  python read_measurement_from_parquet_local.py
      -r 0
      -n <path-to.parquet>
      -t cycling
      --arg-indices
"""

from os.path import isfile
import sys
import xmlhelpy

@xmlhelpy.command(
    name="python read_measurement_from_parquet_local.py",
    version="${VERSION}",
)
@xmlhelpy.option(
    "record",
    char="r",
    param_type=xmlhelpy.Integer,
    required=False,
    description="(Ignored) Kadi record id. Local-only.",
)
@xmlhelpy.option(
    "filename",
    char="n",
    param_type=xmlhelpy.String,
    required=True,
    description="Path to the local Parquet file.",
)
@xmlhelpy.option(
    "datatype",
    char="t",
    default="cycling",
    param_type=xmlhelpy.Choice(["cycling", "static", "impedance"]),
    description=(
        "Measurement type: 'cycling' -> Cycling_Information, "
        "'static' -> Static_Information, 'impedance' -> Impedance_Information."
    ),
)
@xmlhelpy.option(
    "arg-indices",
    char="a",
    is_flag=True,
    description="Replace indices with (0, ..., len(data)-1).",
)
def read_measurement_from_parquet_local(
    record,
    filename,
    datatype,
    arg_indices,
):
    """Local parquet -> EP-BOLFI JSON (stdout)."""
    from ep_bolfi.utility.dataset_formatting import read_parquet_table

    if not isfile(filename):
        raise FileNotFoundError(
            f"Parquet file not found: {filename}\n"
            "This local tool does not download from Kadi."
        )

    data = read_parquet_table(filename, datatype)

    if arg_indices:
        data.indices = [i for i in range(len(data))]

    # Emit JSON on stdout so you can `>` redirect or pipe to next tool
    sys.stdout.write(data.to_json())

if __name__ == "__main__":
    read_measurement_from_parquet_local()
