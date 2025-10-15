# SCRIPTS/helpers.py
from __future__ import annotations
import os, gc, re, subprocess, contextlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


LOG_DIR: Path | None = None

def init_helpers(log_dir: Path):
    """Call this once from your main script to set the log directory."""
    global LOG_DIR
    LOG_DIR = Path(log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _log_files(tag: str):
    if LOG_DIR is None:
        raise RuntimeError("helpers.init_helpers(log_dir) must be called before use.")
    base = LOG_DIR / f"{_ts()}_{tag}"
    return base.with_suffix(".out"), base.with_suffix(".err")

def _print_step(title: str):
    print(f"\n{'-'*88}\n{title}\n{'-'*88}")

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def run_to_files(
    cmd,
    *,
    stdin_path: Path | None = None,
    stdout_path: Path | None = None,
    tag: str = "step",
):
    """Run a command streaming I/O to files (low memory)."""
    out_log, err_log = _log_files(tag)
    stdout_handle = open(stdout_path, "w", encoding="utf-8") if stdout_path else open(out_log, "w", encoding="utf-8")
    stderr_handle = open(err_log, "w", encoding="utf-8")
    stdin_handle = open(stdin_path, "r", encoding="utf-8") if stdin_path else None

    try:
        print("RUN:", " ".join(map(str, cmd)))
        subprocess.run(
            cmd,
            text=True,
            check=True,
            env=os.environ,
            stdin=stdin_handle,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
        )
    except subprocess.CalledProcessError as e:
        try:
            stderr_handle.flush()
            err_tail = Path(err_log).read_text(encoding="utf-8", errors="ignore")[-4000:]
        except Exception:
            err_tail = "<failed to read stderr log>"
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n\n[stderr tail]\n{err_tail}") from e
    finally:
        if stdin_handle: stdin_handle.close()
        stdout_handle.close()
        stderr_handle.close()
        del stdin_handle, stdout_handle, stderr_handle
        gc.collect()

    if stdout_path is None:
        print(f"  stdout → {out_log}")
    print(f"  stderr → {err_log}")

def run_to_single_file_stdout_only(
    cmd,
    *,
    stdin_path: Path | None = None,
    stdout_path: Path,
    stderr_path: Path | None = None,
    unbuffered_python: bool = True,
):
    """stdout → stdout_path (.py), stderr → stderr_path (.err)."""
    ensure_parent(stdout_path)
    if stderr_path is None:
        stderr_path = stdout_path.with_suffix(".err")

    env = os.environ.copy()
    if unbuffered_python:
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")

    with open(stdout_path, "w", encoding="utf-8") as out, \
         open(stderr_path, "w", encoding="utf-8") as err, \
         (open(stdin_path, "r", encoding="utf-8") if stdin_path else contextlib.nullcontext()) as in_h:
        print("RUN:", " ".join(map(str, cmd)))
        try:
            subprocess.run(
                cmd,
                text=True,
                check=True,
                env=env,
                stdin=(in_h if stdin_path else None),
                stdout=out,
                stderr=err,   # warnings go here
                close_fds=True,
            )
        except subprocess.CalledProcessError as e:
            tail = Path(stderr_path).read_text(encoding="utf-8", errors="ignore")[-4000:]
            raise RuntimeError(
                f"Command failed: {' '.join(map(str, cmd))}\n\n[stderr tail]\n{tail}"
            ) from e
    print(f"(step 3) stdout → {stdout_path}")
    print(f"(step 3) stderr → {stderr_path}")

def sanitize_ocv_module(raw_path: Path, *, overwrite: bool = False) -> Path:
    """
    Clean auto-generated ocv_function .py:
    - Comments header/report lines above the function
    - Normalizes unicode subscripts ₀…₉ → 0…9 (e.g., E₀ -> E0)
    - Ensures 'import numpy as np' is present at the top
    Writes <name>.py (if overwrite=False) or overwrites original.
    """
    out_path = raw_path if overwrite else raw_path.with_name(raw_path.stem + ".py")
    src = raw_path.read_text(encoding="utf-8", errors="ignore")

    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    src = src.translate(subs)

    m = re.search(r'^\s*def\s+OCV_Curve_Fit\s*\(', src, flags=re.M)
    if not m:
        raise RuntimeError("Couldn't find 'def OCV_Curve_Fit(' in the generated file.")
    header, rest = src[:m.start()], src[m.start():]
    commented_header = "\n".join("# " + ln for ln in header.splitlines())
    preamble = "# -*- coding: utf-8 -*-\nimport numpy as np\n"
    clean_src = f"{preamble}\n{commented_header}\n\n{rest}"
    out_path.write_text(clean_src, encoding="utf-8")
    print(f"Sanitized file written to: {out_path}")
    return out_path


# --- existing imports and helpers you already have ---


# ... your existing: LOG_DIR, init_helpers, _print_step, ensure_parent,
#     run_to_files, run_to_single_file_stdout_only, sanitize_ocv_module ...

# ---------------- Pipeline Config ----------------
@dataclass
class PipelineConfig:
    PYTHON: str | Path
    REPO_ROOT: Path
    DATA_DIR: Path
    SCRIPTS: Path

    # Step 1
    parquet_path: Path
    json_from_parquet: Path

    # Step 2
    extract_ocv_curve_local_py: Path
    ocv_extraction_basename: Path
    ocv_extract_plot_title: str
    select_segments_arg: str
    split_on_current_direction: str
    positive_current_is_lithiation: str

    # Step 3
    ocv_fit_input_json: Path
    fit_and_plot_ocv_local_py: Path
    ocv_fit_title: str
    soc_key: str
    ocv_key: str
    poly_degree: str
    weight_exp: str
    temp_K: str
    ocv_bounds: str
    soc_bounds: str
    electrode_name: str
    plot_points: str
    fit_domain: str
    jump_penalty: str
    export_lang: str
    extrapolate: str
    derivative_order: str
    fit_kwargs: str

    # Step 5
    data_path_json: Path
    selected_segments_json: Path
    segments_selection: str
    measurement_type: str

    # Step 6
    extract_overpotential_script: Path
    ocv_fit_for_local: Path
    parameters_file: Path
    electrode_choice: str
    current_sign: str
    figure_format: str
    overwrite_fig: str
    overpotentials_out: Path

    # Step 7
    store_overpotential_as_parquet_local_py: Path
    overpotentials_parquet_basename: Path
    parquet_compression_level: int
    parquet_overwrite: bool


# ---------------- Pipeline Steps ----------------
def step1_read_parquet(cfg: PipelineConfig):
    _print_step("STEP 1: read_measurement_from_parquet → JSON (local, streamed)")
    ensure_parent(cfg.json_from_parquet)
    cmd = [
        str(cfg.PYTHON),
        str(cfg.SCRIPTS / "read_measurement_from_parquet_local.py"),
        "-r", "0",
        "-n", str(cfg.parquet_path),
        "-t", "cycling",
        "--arg-indices",
    ]
    run_to_files(cmd, stdout_path=cfg.json_from_parquet, tag="step1")
    print(f"Wrote: {cfg.json_from_parquet}")
    gc.collect()


def step2_extract_ocv(cfg: PipelineConfig):
    _print_step("STEP 2: extract_ocv_curve (local, streamed)")
    cmd = [
        str(cfg.PYTHON),
        str(cfg.extract_ocv_curve_local_py),
        "-n", str(cfg.ocv_extraction_basename),
        "-t", str(cfg.ocv_extract_plot_title),
        "-f", "pdf",
        "-s", str(cfg.select_segments_arg),
        "-c", "0.0",
        "--split-on-current-direction", str(cfg.split_on_current_direction),
        "--positive-current-is-lithiation", str(cfg.positive_current_is_lithiation),
    ]
    run_to_files(cmd, stdin_path=cfg.json_from_parquet, tag="step2")
    gc.collect()


def step3_fit_and_sanitize(cfg: PipelineConfig):
    _print_step("STEP 3: fit_and_plot_ocv (stdout→.py, stderr→.err, then sanitize)")
    if not Path(cfg.ocv_fit_input_json).exists():
        p = Path(cfg.ocv_fit_input_json).parent
        listing = "\n".join("  - " + c.name for c in p.glob("*"))
        raise FileNotFoundError(f"Missing: {cfg.ocv_fit_input_json}\nDirectory listing of {p}:\n{listing}")

    cmd = [
        str(cfg.PYTHON),
        str(cfg.fit_and_plot_ocv_local_py),
        "-n", str(cfg.ocv_fit_input_json),
        "-t", str(cfg.ocv_fit_title),
        "-f", "pdf",
        "-s", str(cfg.soc_key),
        "-o", str(cfg.ocv_key),
        "-a", str(cfg.poly_degree),
        "-z", str(cfg.weight_exp),
        "-k", str(cfg.temp_K),
        "-u", str(cfg.ocv_bounds),
        "-g", str(cfg.soc_bounds),
        "-c", str(cfg.electrode_name),
        "-p", str(cfg.plot_points),
        "-b", str(cfg.fit_domain),
        "-j", str(cfg.jump_penalty),
        "-l", str(cfg.export_lang),
        "-x", str(cfg.extrapolate),
        "-d", str(cfg.derivative_order),
        "-m", str(cfg.fit_kwargs),
    ]

    ocv_function_path = cfg.DATA_DIR / "ocv_function.py"
    ocv_function_err  = ocv_function_path.with_suffix(".err")
    run_to_single_file_stdout_only(cmd, stdout_path=ocv_function_path, stderr_path=ocv_function_err)

    # produce both cleaned copy and overwrite with cleaned content
    sanitize_ocv_module(ocv_function_path, overwrite=False)
    sanitize_ocv_module(ocv_function_path, overwrite=True)
    gc.collect()


def step5_select_segments(cfg: PipelineConfig):
    _print_step("STEP 5: select_measurement_segments (streamed)")
    cmd = [
        str(cfg.PYTHON),
        "-m", "ep_bolfi.kadi_tools.select_measurement_segments",
        cfg.segments_selection,
        "-t", cfg.measurement_type,
    ]
    run_to_files(cmd, stdin_path=cfg.data_path_json, stdout_path=cfg.selected_segments_json, tag="step5")
    print(f"Wrote: {cfg.selected_segments_json}")
    gc.collect()


def step6_extract_overpotential(cfg: PipelineConfig):
    _print_step("STEP 6: extract_overpotential_local (local, streamed)")
    ensure_parent(cfg.overpotentials_out)
    cmd = [
        str(cfg.PYTHON),
        str(cfg.extract_overpotential_script),
        "-n", str(cfg.ocv_fit_for_local),
        "-q", str(cfg.parameters_file),
        "-e", str(cfg.electrode_choice),
        "-c", str(cfg.current_sign),
        "-f", str(cfg.figure_format),
        "--overwrite", str(cfg.overwrite_fig),
    ]
    run_to_files(cmd, stdin_path=cfg.selected_segments_json, stdout_path=cfg.overpotentials_out, tag="step6")
    print(f"Wrote: {cfg.overpotentials_out}")
    gc.collect()


def step7_store_overpotential(cfg: PipelineConfig):
    _print_step("STEP 7: store_overpotential_as_parquet_local (local, streamed)")
    cmd = [
        str(cfg.PYTHON),
        str(cfg.store_overpotential_as_parquet_local_py),
        "-n", str(cfg.overpotentials_parquet_basename),
        "-c", str(cfg.parquet_compression_level),
    ]
    if cfg.parquet_overwrite:
        cmd.append("--overwrite")
    run_to_files(cmd, stdin_path=cfg.overpotentials_out, tag="step7")
    print(f"Wrote: {cfg.overpotentials_parquet_basename.with_suffix('.parquet')}")
    gc.collect()


def run_pipeline(cfg: PipelineConfig):
    step1_read_parquet(cfg)
    step2_extract_ocv(cfg)
    step3_fit_and_sanitize(cfg)
    _print_step("STEP 4: Ensure parameters file is updated")
    step5_select_segments(cfg)
    step6_extract_overpotential(cfg)
    step7_store_overpotential(cfg)
    _print_step("ALL DONE ✅")
