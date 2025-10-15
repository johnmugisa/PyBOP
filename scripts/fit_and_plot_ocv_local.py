# """
# LOCAL-ONLY: Fit an OCV(SOC) curve from a *_extraction.json produced by
# extract_ocv_curve_local.py and write:
#   - a plot (PDF/PNG/etc),
#   - a small Python file exporting OCV_Curve_Fit(SOC),
#   - a JSON with polynomial coefficients.
# No Kadi required.

# Usage (mirrors the Kadi CLI flags you used):
#   python fit_and_plot_ocv_local.py
#     -n <path-to-*_extraction.json>
#     -t "OCV Curve Fit"
#     -f pdf
#     -s "SOC [C]"
#     -o "OCV [V]"
#     -a 3
#     -z 1.0
#     -k 298.15
#     -u "(0.01, 0.95)"
#     -g "(0.0, 1.0)"
#     -c Negative
#     -p false
#     -b "(0.001, 0.999)"
#     -j 2
#     -l python
#     -x false
#     -d "2"
#     -m "{}"
# """

# import json, math, ast
# import xmlhelpy
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# @xmlhelpy.command(name="python fit_and_plot_ocv_local.py", version="${VERSION}")
# @xmlhelpy.option("input",  char="n", param_type=xmlhelpy.String, required=True, description="*_extraction.json path")
# @xmlhelpy.option("title",  char="t", param_type=xmlhelpy.String, default="OCV Curve Fit")
# @xmlhelpy.option("format", char="f", param_type=xmlhelpy.Choice(
#     ['eps','jpg','jpeg','pdf','pgf','png','ps','raw','rgba','svg','svgz','tif','tiff'],
#     case_sensitive=True), default="pdf")
# @xmlhelpy.option("soc-key", char="s", param_type=xmlhelpy.String, default="SOC [C]")
# @xmlhelpy.option("ocv-key", char="o", param_type=xmlhelpy.String, default="OCV [V]")
# @xmlhelpy.option("degree", char="a", param_type=xmlhelpy.Integer, default=3, description="Polynomial degree")
# @xmlhelpy.option("weight-exp", char="z", param_type=xmlhelpy.Float, default=1.0,
#                  description="Optional weighting exponent (heuristic)")
# @xmlhelpy.option("temp-K", char="k", param_type=xmlhelpy.Float, default=298.15)
# @xmlhelpy.option("ocv-bounds", char="u", param_type=xmlhelpy.String, default="(0.01, 0.95)")
# @xmlhelpy.option("soc-bounds", char="g", param_type=xmlhelpy.String, default="(0.0, 1.0)")
# @xmlhelpy.option("electrode", char="c", param_type=xmlhelpy.String, default="Negative")
# @xmlhelpy.option("plot-points", char="p", param_type=xmlhelpy.String, default="false")
# @xmlhelpy.option("fit-domain", char="b", param_type=xmlhelpy.String, default="(0.001, 0.999)")
# @xmlhelpy.option("jump-penalty", char="j", param_type=xmlhelpy.Integer, default=2)
# @xmlhelpy.option("export-lang", char="l", param_type=xmlhelpy.String, default="python")
# @xmlhelpy.option("extrapolate", char="x", param_type=xmlhelpy.String, default="false")
# @xmlhelpy.option("derivative-order", char="d", param_type=xmlhelpy.String, default="2")
# @xmlhelpy.option("fit-kwargs", char="m", param_type=xmlhelpy.String, default="{}")
# def fit_and_plot_ocv_local(input, title, format, soc_key, ocv_key, degree,
#                            weight_exp, temp_K, ocv_bounds, soc_bounds, electrode,
#                            plot_points, fit_domain, jump_penalty, export_lang,
#                            extrapolate, derivative_order, fit_kwargs):
#     in_path = Path(input)
#     if not in_path.exists():
#         raise FileNotFoundError(in_path)

#     with open(in_path, "r", encoding="utf-8") as f:
#         D = json.load(f)
#     SOC = np.asarray(D.get(soc_key, []), dtype=float)
#     OCV = np.asarray(D.get(ocv_key, []), dtype=float)
#     if SOC.size == 0 or OCV.size == 0 or SOC.size != OCV.size:
#         raise ValueError("Invalid extraction JSON: missing or mismatched SOC/OCV arrays.")

#     # Parse bounds
#     ocv_lo, ocv_hi = ast.literal_eval(ocv_bounds)
#     soc_lo, soc_hi = ast.literal_eval(soc_bounds)
#     dom_lo, dom_hi = ast.literal_eval(fit_domain)

#     # Filter by bounds
#     mask = (SOC >= soc_lo) & (SOC <= soc_hi) & (OCV >= ocv_lo) & (OCV <= ocv_hi)
#     SOC_fit = SOC[mask]
#     OCV_fit = OCV[mask]
#     if SOC_fit.size < degree + 1:
#         raise ValueError(f"Not enough points ({SOC_fit.size}) to fit degree {degree} polynomial.")

#     # Weights (simple heuristic): down-weight near the edges of fit domain if weight_exp > 1
#     # distance to nearest boundary in normalized domain:
#     norm_soc = (SOC_fit - dom_lo) / max(1e-12, (dom_hi - dom_lo))
#     dist = np.minimum(norm_soc, 1 - norm_soc).clip(1e-6, 1.0)
#     w = dist ** float(weight_exp)

#     # Polynomial fit
#     coeffs = np.polyfit(SOC_fit, OCV_fit, deg=int(degree), w=w)

#     # Build outputs
#     base = in_path.with_suffix("")  # remove .json
#     out_prefix = str(base).replace("_extraction", "_fit")
#     plot_path = f"{out_prefix}.{format}"
#     coef_json = f"{out_prefix}_coefficients.json"
#     export_py = f"{out_prefix}_function.py"

#     # Plot
#     xs = np.linspace(dom_lo, dom_hi, 800)
#     ys = np.polyval(coeffs, xs)

#     fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4), constrained_layout=True)
#     if str(plot_points).lower() == "true":
#         ax.scatter(SOC_fit, OCV_fit, s=12, alpha=0.8, label="Data")
#     ax.plot(xs, ys, label=f"poly deg {degree}")
#     ax.set_xlabel(soc_key)
#     ax.set_ylabel(ocv_key)
#     ax.set_title(title)
#     ax.legend()
#     fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.0)

#     # Save coefficients & export function
#     with open(coef_json, "w", encoding="utf-8") as f:
#         json.dump({"degree": int(degree), "coefficients_highest_power_first": coeffs.tolist()}, f, indent=2)

#     code = f'''# Auto-generated from {in_path.name}
# # Polynomial degree {int(degree)}. numpy.polyval-compatible coeffs (highest power first).
# import numpy as np
# COEFFS = np.array({coeffs.tolist()}, dtype=float)

# def OCV_Curve_Fit(SOC):
#     # SOC can be scalar or numpy array in [0,1]
#     return np.polyval(COEFFS, SOC)
# '''
#     with open(export_py, "w", encoding="utf-8") as f:
#         f.write(code)

#     print(json.dumps({
#         "fit_plot": plot_path,
#         "fit_coefficients_json": coef_json,
#         "export_function_py": export_py,
#         "degree": int(degree),
#         "n_points": int(SOC_fit.size),
#         "soc_bounds": [soc_lo, soc_hi],
#         "ocv_bounds": [ocv_lo, ocv_hi],
#         "fit_domain": [dom_lo, dom_hi],
#     }, indent=2))

# if __name__ == "__main__":
#     fit_and_plot_ocv_local()


"""
LOCAL-ONLY replacement for:
  python -m ep_bolfi.kadi_tools.fit_and_plot_ocv

Reads a JSON (from disk) that contains OCV data and fits the Birkl2015 model
(via ep_bolfi.utility.visualization.fit_and_plot_OCV). Writes:
  - <prefix>_fit_log.json
  - <prefix>_fit_plot.<format>
  - <prefix>_with_<electrode>_soc.json   (only if --assign-fitted-soc-to is given)

NO Kadi calls. The options -r/--input-record and -e/--output-record are accepted
for CLI compatibility but ignored.
"""

from ast import literal_eval
import json
from os.path import isfile
import xmlhelpy

@xmlhelpy.command(
    name='python fit_and_plot_ocv_local_birkl.py',
    version='${VERSION}'
)
@xmlhelpy.option(
    'input-record',
    char='r',
    param_type=xmlhelpy.Integer,
    required=False,
    description="(Ignored) Kadi record ID where input is stored. Local-only."
)
@xmlhelpy.option(
    'filename',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description="Path to the input JSON file on disk."
)
@xmlhelpy.option(
    'output-record',
    char='e',
    param_type=xmlhelpy.Integer,
    required=False,
    description="(Ignored) Kadi record ID to upload results to. Local-only."
)
@xmlhelpy.option(
    'title',
    char='t',
    default=None,
    param_type=xmlhelpy.String,
    description="Title of the plot. Defaults to the file name (prefix)."
)
@xmlhelpy.option(
    'format',
    char='f',
    default='pdf',
    param_type=xmlhelpy.Choice(
        ['eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba',
         'svg', 'svgz', 'tif', 'tiff'],
        case_sensitive=True
    ),
    description="Format of generated image file."
)
@xmlhelpy.option(
    'soc-key',
    char='s',
    default='SOC [C]',
    param_type=xmlhelpy.String,
    description="Key in the json dictionary for the SOC values."
)
@xmlhelpy.option(
    'ocv-key',
    char='o',
    default='OCV [V]',
    param_type=xmlhelpy.String,
    description="Key in the json dictionary for the OCV values."
)
@xmlhelpy.option(
    'phases',
    char='a',
    default=4,
    param_type=xmlhelpy.Integer,
    description="Number of phases of the OCV model. More are more accurate."
)
@xmlhelpy.option(
    'charge-number',
    char='z',
    default=1.0,
    param_type=xmlhelpy.Float,
    description="The charge number of the electrode interface reaction."
)
@xmlhelpy.option(
    'temperature',
    char='k',
    default=298.15,
    param_type=xmlhelpy.Float,
    description="Temperature at which the OCV got measured."
)
@xmlhelpy.option(
    'soc-range-bounds',
    char='u',
    default='(0.2, 0.8)',
    param_type=xmlhelpy.String,
    description=(
        "2-tuple giving the lower maximum and upper minimum SOC range to be "
        "considered in the automatic data SOC range determination."
    )
)
@xmlhelpy.option(
    'soc-range-limits',
    char='g',
    default='(0.0, 1.0)',
    param_type=xmlhelpy.String,
    description=(
        "Optional hard lower and upper bounds for the SOC correction from "
        "the left and the right side, respectively, as a 2-tuple. Use it "
        "if you know that your OCV data is incomplete and by how much. "
        "Has to be inside (0.0, 1.0). Set to (0.0, 1.0) to allow the "
        "SOC range estimation to assign datapoints to the asymptotes."
    )
)
@xmlhelpy.option(
    'assign-fitted-soc-to',
    char='c',
    default=None,
    param_type=xmlhelpy.Choice(
        [None, 'Positive', 'Negative'],
        case_sensitive=False,
    ),
    description=(
        "Set this to 'Positive' or 'Negative' to apply the fitted SOC range "
        "on the data and assign the result to one of the two electrodes."
    )
)
@xmlhelpy.option(
    'flip-soc-convention',
    char='p',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "Set to True if assigned SOCs shall go in the other direction. "
        "'soc-range' arguments always work as if this was set to False. "
        "Flips the SOCs by subtracting them from 1."
    )
)
@xmlhelpy.option(
    'spline-soc-range',
    char='b',
    default='(0.01, 0.99)',
    param_type=xmlhelpy.String,
    description=(
        "2-tuple giving the SOC range in which the SOC(OCV) model function "
        "gets inverted by a smoothing spline interpolation."
    )
)
@xmlhelpy.option(
    'spline-order',
    char='j',
    default=2,
    param_type=xmlhelpy.Integer,
    description=(
        "Order of the aforementioned smoothing spline. Setting it to 0 "
        "only fits and plots the OCV model."
    )
)
@xmlhelpy.option(
    'spline-print',
    char='l',
    default=None,
    param_type=xmlhelpy.String,
    description=(
        "If set to either 'python' or 'matlab', a string representation of "
        "the smoothing spline gets appended to the results file."
    )
)
@xmlhelpy.option(
    'normalized-xaxis',
    char='x',
    default=False,
    param_type=xmlhelpy.Bool,
    description=(
        "If True, the x-axis gets rescaled to match the asymptotes of the OCV "
        "fit function at 0 and 1."
    )
)
@xmlhelpy.option(
    'distance-order',
    char='d',
    default='2',
    param_type=xmlhelpy.String,
    description=(
        "The order of the norm of the vector of the distances between OCV "
        "data and OCV model. Default is 2, i.e., the Euclidean norm. "
        "1 sets it to absolute distance, and float('inf') sets it to "
        "maximum distance."
    )
)
@xmlhelpy.option(
    'minimize-options',
    char='m',
    default="{}",
    param_type=xmlhelpy.String,
    description=(
        "Dictionary that gets passed to scipy.optimize.minimize with the "
        "method 'trust-constr'. See scipy.optimize.show_options with the "
        "arguments 'minimize' and 'trust-constr' for details."
    )
)
@xmlhelpy.option(
    'overwrite',
    char='w',
    default=False,
    param_type=xmlhelpy.Bool,
    description="(Ignored) Local-only; files are written beside input."
)
@xmlhelpy.option(
    'display',
    char='v',
    is_flag=True,
    description="Toggle to display the plot."
)
def fit_and_plot_ocv(
    input_record,
    filename,
    output_record,
    title,
    format,
    soc_key,
    ocv_key,
    phases,
    charge_number,
    temperature,
    soc_range_bounds,
    soc_range_limits,
    assign_fitted_soc_to,
    flip_soc_convention,
    spline_soc_range,
    spline_order,
    spline_print,
    normalized_xaxis,
    distance_order,
    minimize_options,
    overwrite,
    display
):
    """Local-only Birkl2015 OCV fit and plot."""
    from ep_bolfi.utility.visualization import fit_and_plot_OCV
    import matplotlib.pyplot as plt

    file_prefix = filename.split('.')[0]
    if title is None:
        title = file_prefix

    # Local-only: the file must exist on disk
    if not isfile(filename):
        raise FileNotFoundError(
            f"Input file not found: {filename}\n"
            "This local tool does not download from Kadi; "
            "write the extraction JSON locally and point -n to it."
        )

    with open(filename, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        socs, ocvs = zip(*sorted(zip(json_data[soc_key], json_data[ocv_key])))

    # Ensure conventional orientation (inverted=True expects OCV decreasing with SOC)
    if max(socs) < 0.0:
        socs = [-entry for entry in socs]
    if max(ocvs) < 0.0:
        ocvs = [-entry for entry in ocvs]
    if not ocvs[0] > ocvs[-1]:
        socs, ocvs = zip(*reversed(list(zip(socs, ocvs))))

    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4))
    ocv_fit_result = fit_and_plot_OCV(
        ax,
        socs,
        ocvs,
        title,
        phases=phases,
        z=charge_number,
        T=temperature,
        SOC_range_bounds=literal_eval(soc_range_bounds),
        SOC_range_limits=literal_eval(soc_range_limits),
        spline_SOC_range=literal_eval(spline_soc_range),
        spline_order=spline_order,
        spline_print=spline_print,
        parameters_print=True,
        inverted=True,
        info_accuracy=True,
        normalized_xaxis=normalized_xaxis,
        distance_order=float(distance_order),
        minimize_options=literal_eval(minimize_options),
    )

    # Save outputs locally
    with open(file_prefix + '_fit_log.json', 'w', encoding='utf-8') as f:
        f.write(ocv_fit_result.to_json())

    ax.set_xlabel(soc_key)
    ax.set_ylabel(ocv_key)
    fig.tight_layout()
    fig.savefig(
        file_prefix + '_fit_plot.' + format,
        bbox_inches='tight',
        pad_inches=0.0
    )
    if display:
        plt.show()

    # Optional assignment of SOCs back into the JSON (local file)
    if assign_fitted_soc_to is not None:
        true_SOCs = list(
            ocv_fit_result.SOC_range[0]
            + (ocv_fit_result.SOC - ocv_fit_result.SOC[0])
            / (ocv_fit_result.SOC[-1] - ocv_fit_result.SOC[0])
            * (ocv_fit_result.SOC_range[1] - ocv_fit_result.SOC_range[0])
        )
        true_SOCs = true_SOCs[::-1]
        if flip_soc_convention:
            true_SOCs = [1 - t for t in true_SOCs]
        json_data[
            assign_fitted_soc_to.capitalize() + " electrode SOC [-]"
        ] = true_SOCs
        with open(
            file_prefix + '_with_' + assign_fitted_soc_to.lower()
            + '_soc.json',
            'w', encoding='utf-8'
        ) as f:
            json.dump(json_data, f)

if __name__ == '__main__':
    fit_and_plot_ocv()
