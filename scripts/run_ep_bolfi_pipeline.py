# ################################## How to run###################
# # Ensure your environment has ep_bolfi, pybamm, etc.

# # If Steps 2–3 need Kadi access, make sure:

# $env:KADI_URL = "https://kadi-dlr.hiu-batteries.de/api"
# $env:KADI_TOKEN = "pat_2ee09f68aa9dad831027b6def0ed0bcd96c4ea5795478e1a"

#################
# Remove-Item Env:KADI_TOKEN -ErrorAction SilentlyContinue
# $env:KADI_TOKEN = "pat_2ee09f68aa9dad831027b6def0ed0bcd96c4ea5795478e1a"
# python -c "import os; print((os.environ['KADI_TOKEN'])[:6], len(os.environ['KADI_TOKEN']))"
###################
# # Update any paths/IDs at the top of the script, then:

# # python C:\path\to\run_ep_bolfi_pipeline.py

# ###################################

################################## How to run ###################
# Ensure your environment has ep_bolfi, pybamm, etc.
#
# If Steps 2–3 need Kadi access, make sure in THIS PowerShell window:
#   $env:KADI_URL = "https://kadi-dlr.hiu-batteries.de/api"
#   $env:KADI_TOKEN = "pat_xxxxx"   # no angle brackets
#
# Then:
#   python C:\path\to\run_ep_bolfi_pipeline.py
#################################################################

import os
import subprocess
import sys
from pathlib import Path

PYTHON = "python"  # or r"C:\Users\mugi_jo\AppData\Local\Programs\Python\Python312\python.exe"

# === TOGGLES ======================================================
USE_LOCAL_STEP2 = True   # use your extract_ocv_curve_local.py
USE_LOCAL_STEP3 = True  # set True if you switch to a local fit script later
# =================================================================

# ---------- CONFIG (edit these) ----------
# Step 1
# parquet_path = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\6813_puls_data_delith_multirowgroup.parquet")
parquet_path = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\6813_puls_data_delith_multirowgroup.parquet")
# json_from_parquet = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\6813_puls_data_delith_multirowgroup.json")
json_from_parquet = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\6813_puls_data_delith_multirowgroup.json")
# Step 2
extract_ocv_curve_local_py = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\extract_ocv_curve_local.py")
# ocv_extraction_basename = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\MyOutputFileName")  # no extension
ocv_extraction_basename = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\MyOutputFileName") 
ocv_extract_record_id = "10513"          # only used if USE_LOCAL_STEP2=False
ocv_extract_plot_title = "My Plot Title"
select_segments_arg = "[(1, None, 2)]"   # literal string
split_on_current_direction = "false"
positive_current_is_lithiation = "true"
overwrite_plots = "false"

# Step 3
ocv_fit_input_json = ocv_extraction_basename.with_name(ocv_extraction_basename.name + "_extraction.json")
fit_and_plot_ocv_local_py = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\fit_and_plot_ocv_local.py")  # only if USE_LOCAL_STEP3=True
ocv_fit_export_record = "10513"          # only used if USE_LOCAL_STEP3=False
ocv_fit_title = "OCV Curve Fit"
soc_key = "SOC [C]"
ocv_key = "OCV [V]"
poly_degree = "3"
weight_exp = "1.0"
temp_K = "298.15"
ocv_bounds = "(0.01, 0.95)"
soc_bounds = "(0.0, 1.0)"
electrode_name = "Negative"      
plot_points = "false"
fit_domain = "(0.001, 0.999)"
jump_penalty = "2"
export_lang = "python"
extrapolate = "false"
derivative_order = "2"
fit_kwargs = "{}"
write_overwrite = "false"
verbose_flag = True

# Step 4 (manual): ensure your parameters file is updated

# Step 5
# data_four_json = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\6813_puls_data_delith_multirowgroup.json")
# selected_segments_four = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\selected_segments.json")
# selected_segments_thirteen = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\selected_segments.json")
data_four_json = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\6813_puls_data_delith_multirowgroup.json")
selected_segments_four = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\selected_segments.json")
selected_segments_thirteen = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\selected_segments.json")
segments_selection = "[(2, None)]"
measurement_type = "cycling"

# Step 6
extract_overpotential_script = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\extract_overpotential_local.py")
ocv_fit_for_local = ocv_fit_input_json
parameters_file = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\customcells_3262_lithiation.py")   ###these are posive based parameters (although for graphite)
electrode_choice = "positive"       ##must be updated depending on the parameter naming positive or negative based parameters
current_sign = "-1"  #-1
figure_format = "pdf"
overwrite_fig = "true"
# overpotentials_out = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\overpotentials.json")
overpotentials_out = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\overpotentials.json")
# ---------- helpers ----------
def require_kadi_env(allow_prompt=True):
    """Only needed if a step talks to Kadi (i.e., USE_LOCAL_* is False)."""
    if USE_LOCAL_STEP2 and USE_LOCAL_STEP3:
        return  # nothing needs Kadi
    import getpass
    url = os.environ.get("KADI_URL", "")
    tok = os.environ.get("KADI_TOKEN", "")
    if not url:
        url = "https://kadi-dlr.hiu-batteries.de/api"
        os.environ["KADI_URL"] = url
    if not tok and allow_prompt:
        print("KADI_TOKEN not found — enter it now (input hidden):")
        tok = getpass.getpass("KADI_TOKEN: ").strip()
    if not tok:
        raise RuntimeError('KADI_TOKEN not set. In PowerShell:  $env:KADI_TOKEN = "pat_xxxxx"')
    if "<" in tok or ">" in tok:
        raise RuntimeError("KADI_TOKEN still contains <…>. Set it without angle brackets.")
    os.environ["KADI_TOKEN"] = tok
    print(f"Kadi env OK ✅ URL={url}, TOKEN={tok[:6]}...{tok[-4:]}")

def assert_kadi_access(test_record_id: str):
    """Fail fast if we cannot access a record (only called when Kadi is used)."""
    from kadi_apy.lib.core import KadiManager, Record
    url = os.environ.get("KADI_URL", "")
    tok = os.environ.get("KADI_TOKEN", "")
    print(f"[Kadi check] URL={url}  token_len={len(tok)}  token_head={tok[:6]}...")
    try:
        m = KadiManager()
        Record(m, id=int(test_record_id), create=False)
        print(f"[Kadi check] ✅ Can access record {test_record_id}")
    except Exception as e:
        print(f"[Kadi check] ❌ Cannot access record {test_record_id}: {e}")
        raise

def run(cmd, **kwargs):
    print("\n" + "-"*80)
    print("RUN:", " ".join(map(str, cmd)))
    print("-"*80)
    return subprocess.run(cmd, text=True, check=True, env=os.environ, **kwargs)

def pipe_json_to(cmd, json_path: Path):
    data = json_path.read_text(encoding="utf-8", errors="ignore")
    return run(cmd, input=data)

# Step 7
overpotentials_out = Path(r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\data_thirteen\delith\overpotentials.json")
overpotentials_parquet_basename = overpotentials_out.with_suffix("")  # => ...\overpotentials
parquet_compression_level = 22
parquet_overwrite = True

# ---------- pipeline ----------
def main():
    # Only require Kadi if a step uses it
    if not (USE_LOCAL_STEP2 and USE_LOCAL_STEP3):
        require_kadi_env()
        # Step 2 uses Kadi only when USE_LOCAL_STEP2=False
        if not USE_LOCAL_STEP2:
            assert_kadi_access(ocv_extract_record_id)
        # Step 3 uses Kadi only when USE_LOCAL_STEP3=False
        if not USE_LOCAL_STEP3:
            assert_kadi_access(ocv_fit_export_record)

    try:
        # STEP 1: parquet -> json
        # print("\nSTEP 1: read_measurement_from_parquet → JSON")
        # cmd1 = [
        #     PYTHON, "-m", "ep_bolfi.kadi_tools.read_measurement_from_parquet",
        #     "-r", "0",
        #     "-n", str(parquet_path),
        #     "-t", "cycling",
        #     "--arg-indices",
        # ]
        # res1 = subprocess.run(cmd1, text=True, capture_output=True, check=True, env=os.environ)
        # json_from_parquet.write_text(res1.stdout, encoding="utf-8")
        # print(f"Wrote: {json_from_parquet}")
        # STEP 1: parquet -> json (local)
        print("\nSTEP 1: read_measurement_from_parquet → JSON (local)")
        cmd1 = [
            PYTHON, r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\read_measurement_from_parquet_local.py",
            "-r", "0",  # ignored locally
            "-n", str(parquet_path),
            "-t", "cycling",
            "--arg-indices",
        ]
        res1 = subprocess.run(cmd1, text=True, capture_output=True, check=True, env=os.environ)
        json_from_parquet.write_text(res1.stdout, encoding="utf-8")
        print(f"Wrote: {json_from_parquet}")


        # STEP 2: extract_ocv_curve
        print("\nSTEP 2: extract_ocv_curve")
        if USE_LOCAL_STEP2:
            # local script (no Kadi)
            cmd2 = [
                PYTHON, str(extract_ocv_curve_local_py),
                "-n", str(ocv_extraction_basename),
                "-t", str(ocv_extract_plot_title),
                "-f", "pdf",
                "-s", str(select_segments_arg),
                "-c", "0.0",
                "--split-on-current-direction", str(split_on_current_direction),
                "--positive-current-is-lithiation", str(positive_current_is_lithiation),
            ]
        else:
            # Kadi CLI
            cmd2 = [
                PYTHON, "-m", "ep_bolfi.kadi_tools.extract_ocv_curve",
                "-r", str(ocv_extract_record_id),
                "-n", str(ocv_extraction_basename),
                "-t", str(ocv_extract_plot_title),
                "-f", "pdf",
                "-s", str(select_segments_arg),
                "-c", "0.0",
                "--split-on-current-direction", str(split_on_current_direction),
                "--positive-current-is-lithiation", str(positive_current_is_lithiation),
                "--overwrite", str(overwrite_plots),
            ]
        pipe_json_to(cmd2, json_from_parquet)

        # STEP 3: fit_and_plot_ocv
        if not Path(ocv_fit_input_json).exists():
            print(f"❌ Step 3 input JSON not found: {ocv_fit_input_json}")
            p = Path(ocv_fit_input_json).parent
            print("Directory listing:", p)
            for child in p.glob("*"):
                print("  -", child.name)
            raise FileNotFoundError(str(ocv_fit_input_json))

        print("\nSTEP 3: fit_and_plot_ocv")
        if USE_LOCAL_STEP3:
            # You’ll need a local script; call it like this:
            cmd3 = [
                PYTHON, str(fit_and_plot_ocv_local_py),
                "-n", str(ocv_fit_input_json),
                "-t", str(ocv_fit_title),
                "-f", "pdf",
                "-s", str(soc_key),
                "-o", str(ocv_key),
                "-a", str(poly_degree),
                "-z", str(weight_exp),
                "-k", str(temp_K),
                "-u", str(ocv_bounds),
                "-g", str(soc_bounds),
                "-c", str(electrode_name),
                "-p", str(plot_points),
                "-b", str(fit_domain),
                "-j", str(jump_penalty),
                "-l", str(export_lang),
                "-x", str(extrapolate),
                "-d", str(derivative_order),
                "-m", str(fit_kwargs),
            ]
        else:
            # Kadi CLI
            cmd3 = [
                PYTHON, "-m", "ep_bolfi.kadi_tools.fit_and_plot_ocv",
                "-r", str(ocv_extract_record_id),
                "-n", str(ocv_fit_input_json),
                "-e", str(ocv_fit_export_record),
                "-t", str(ocv_fit_title),
                "-f", "pdf",
                "-s", str(soc_key),
                "-o", str(ocv_key),
                "-a", str(poly_degree),
                "-z", str(weight_exp),
                "-k", str(temp_K),
                "-u", str(ocv_bounds),
                "-g", str(soc_bounds),
                "-c", str(electrode_name),
                "-p", str(plot_points),
                "-b", str(fit_domain),
                "-j", str(jump_penalty),
                "-l", str(export_lang),
                "-x", str(extrapolate),
                "-d", str(derivative_order),
                "-m", str(fit_kwargs),
                "-w", str(write_overwrite),
            ]
            if verbose_flag:
                cmd3.append("-v")
        run(cmd3)

        # STEP 4
        print("\nSTEP 4: Make sure your parameters file is updated. Continuing...")

        # STEP 5a: select_measurement_segments (data_four)
        print("\nSTEP 5a: select_measurement_segments (data_four)")
        cmd5a = [PYTHON, "-m", "ep_bolfi.kadi_tools.select_measurement_segments", segments_selection, "-t", measurement_type]
        res5a = subprocess.run(cmd5a, text=True,
                               input=data_four_json.read_text(encoding="utf-8", errors="ignore"),
                               capture_output=True, check=True, env=os.environ)
        selected_segments_four.write_text(res5a.stdout, encoding="utf-8")
        print(f"Wrote: {selected_segments_four}")

        # STEP 5b: select_measurement_segments (data_thirteen)
        print("\nSTEP 5b: select_measurement_segments (data_thirteen)")
        cmd5b = [PYTHON, "-m", "ep_bolfi.kadi_tools.select_measurement_segments", segments_selection, "-t", measurement_type]
        res5b = subprocess.run(cmd5b, text=True,
                               input=Path(json_from_parquet).read_text(encoding="utf-8", errors="ignore"),
                               capture_output=True, check=True, env=os.environ)
        selected_segments_thirteen.write_text(res5b.stdout, encoding="utf-8")
        print(f"Wrote: {selected_segments_thirteen}")

        # STEP 6: extract_overpotential_local.py (stdin from selected_segments_thirteen)
        print("\nSTEP 6: extract_overpotential_local (local-only)")
        cmd6 = [
            PYTHON, str(extract_overpotential_script),
            "-n", str(ocv_fit_for_local),
            "-q", str(parameters_file),
            "-e", str(electrode_choice),
            "-c", str(current_sign),
            "-f", str(figure_format),
            "--overwrite", str(overwrite_fig),
        ]
        res6 = subprocess.run(cmd6, text=True,
                              input=selected_segments_thirteen.read_text(encoding="utf-8", errors="ignore"),
                              capture_output=True, check=True, env=os.environ)
        overpotentials_out.write_text(res6.stdout, encoding="utf-8")
        print(f"Wrote: {overpotentials_out}")

        print("\nSTEP 7: store_overpotential_as_parquet_local")

        cmd7 = [
            PYTHON, r"C:\Users\mugi_jo\Documents\DLR_PROJECTS\ruphay_data\store_overpotential_as_parquet_local.py",
            "-n", str(overpotentials_parquet_basename),          # e.g. ...\data_thirteen\overpotentials
            "-c", str(parquet_compression_level),
        ]
        if parquet_overwrite:
            cmd7.append("--overwrite")    # click flag: no value

        res7 = subprocess.run(
            cmd7,
            text=True,
            input=overpotentials_out.read_text(encoding="utf-8", errors="ignore"),  # pipe Step 6 JSON
            capture_output=True,
            check=True,
            env=os.environ,
        )

        print(res7.stdout if res7.stdout.strip()
            else f"Wrote: {str(overpotentials_parquet_basename.with_suffix('.parquet'))}")

        print("\nALL DONE")

    except subprocess.CalledProcessError as e:
        print("\nA command failed.")
        print("COMMAND:", " ".join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd))
        print("RETURN CODE:", e.returncode)
        if e.stdout:
            print("\n--- STDOUT ---\n" + e.stdout)
        if e.stderr:
            print("\n--- STDERR ---\n" + e.stderr)
        sys.exit(e.returncode)
    except Exception as ex:
        print("\nUnexpected error:", ex)
        sys.exit(1)



if __name__ == "__main__":
    main()
