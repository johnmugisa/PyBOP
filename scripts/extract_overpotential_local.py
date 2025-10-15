"""
Reads in a ``Cycling_Information`` json representation and applies a
PyBaMM model parameter set to subtract the OCP curve(s) from it, leaving
us with only the overpotential. (LOCAL-ONLY VERSION)
"""

import json
import sys
import os
import importlib.util
import matplotlib.pyplot as plt
from os import linesep
from os.path import isfile
from scipy.optimize import root_scalar
import xmlhelpy


def ocv_mismatch(soc, ocv, parameters, electrode='both', voltage_sign=0):
    if electrode != 'both':
        if voltage_sign == 0:
            voltage_sign = 1 if electrode == "positive" else -1
        return (
            ocv
            - voltage_sign * parameters[
                electrode.capitalize() + " electrode OCP [V]"
            ](soc)
        )
    else:
        return (
            ocv
            - parameters["Positive electrode OCP [V]"](soc)
            + parameters["Negative electrode OCP [V]"](soc)
        )


def transform_to_unity_interval(segment):
    return [
        (s - segment[0]) / (segment[-1] - segment[0])
        for s in segment
    ]


@xmlhelpy.command(
    name='python extract_overpotential_local.py',
    version='${VERSION}'
)
@xmlhelpy.option(
    'ocv-file',
    char='n',
    param_type=xmlhelpy.String,
    required=True,
    description="Local JSON file with OCV fit (from fit_and_plot_ocv)."
)
@xmlhelpy.option(
    'parameters-file',
    char='q',
    param_type=xmlhelpy.String,
    required=True,
    description=(
        "Local Python file with model parameters. Must define:\n"
        " - parameters (dict)\n"
        " - negative_SOC_from_cell_SOC (callable, if electrode='both')\n"
        " - positive_SOC_from_cell_SOC (callable, if electrode='both')"
    )
)
@xmlhelpy.option(
    'source-index',
    char='s',
    param_type=xmlhelpy.Float,
    default=float('inf'),
    description="Index of data in OCV file to use as GITT start."
)
@xmlhelpy.option(
    'adjust-for-ocv-mismatch',
    char='a',
    param_type=xmlhelpy.Bool,
    default=True,
    description="Subtract OCV imperfections in addition to fit."
)
@xmlhelpy.option(
    'title',
    char='t',
    default=None,
    param_type=xmlhelpy.String,
    description="Title of the plot."
)
@xmlhelpy.option(
    'electrode',
    char='e',
    default='both',
    param_type=xmlhelpy.Choice(
        ['positive', 'negative', 'both'], case_sensitive=True
    ),
    description="Which OCP(s) to remove."
)
@xmlhelpy.option(
    'current-sign',
    char='c',
    default=0,
    param_type=xmlhelpy.Integer,
    description="Sign convention for SOC update (only if electrode != 'both')."
)
@xmlhelpy.option(
    'voltage-sign',
    char='u',
    default=0,
    param_type=xmlhelpy.Integer,
    description="Sign convention for voltage (only if electrode != 'both')."
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
    'overwrite',
    char='w',
    default=False,
    param_type=xmlhelpy.Bool,
    description="Overwrite existing files."
)
@xmlhelpy.option(
    'display',
    char='v',
    is_flag=True,
    description="Display the plot interactively."
)
def extract_overpotential(
    ocv_file,
    parameters_file,
    source_index,
    adjust_for_ocv_mismatch,
    title,
    electrode,
    current_sign,
    voltage_sign,
    format,
    overwrite,
    display,
):
    """Local-only overpotential extraction."""
    from ep_bolfi.utility.dataset_formatting import Cycling_Information
    from ep_bolfi.utility.preprocessing import (
        subtract_OCV_curve_from_cycles, subtract_both_OCV_curves_from_cycles
    )
    from ep_bolfi.utility.visualization import plot_measurement

    file_prefix = ocv_file.split('.')[0]
    if title is None:
        title = file_prefix

    # Load measurement JSON from stdin
    try:
        data = Cycling_Information.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError(
            "No measurement file (or a corrupted one) was passed/piped."
        )

    # Load OCV fit
    if not isfile(ocv_file):
        raise FileNotFoundError(f"OCV file not found: {ocv_file}")
    with open(ocv_file, 'r') as f:
        ocv_data = json.load(f)

    # Load parameters from local Python file
    param_path = os.path.abspath(parameters_file)
    if not isfile(param_path):
        raise FileNotFoundError(f"Parameters file not found: {param_path}")
    spec = importlib.util.spec_from_file_location("local_parameter_file", param_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parameters = mod.parameters
    if electrode == "both":
        positive_SOC_from_cell_SOC = mod.positive_SOC_from_cell_SOC
        negative_SOC_from_cell_SOC = mod.negative_SOC_from_cell_SOC

    # SOC setup
    if source_index == float('inf'):
        source_index = data.indices[0] - 1

    initial_socs = {
        "Initial concentration in " + e + " electrode": None
        for e in ["negative", "positive"]
    }
    for e in ["negative", "positive"]:
        key = e.capitalize() + " electrode SOC [-]"
        if key in ocv_data.keys():
            initial_socs["Initial concentration in " + e + " electrode"] = (
                ocv_data[key][ocv_data['indices'].index(source_index)]
            )

    # Subtract OCV
    if electrode != 'both':
        data.voltages, returned_SOCs = subtract_OCV_curve_from_cycles(
            data,
            parameters,
            starting_SOC=initial_socs["Initial concentration in " + electrode + " electrode"],
            electrode=electrode,
            current_sign=current_sign,
            voltage_sign=voltage_sign
        )
    else:
        if initial_socs["Initial concentration in positive electrode"] is not None:
            starting_SOC = root_scalar(
                lambda s: positive_SOC_from_cell_SOC(s)
                - initial_socs["Initial concentration in positive electrode"],
                method='toms748',
                bracket=[0, 1],
                x0=0.5
            ).root
        elif initial_socs["Initial concentration in negative electrode"] is not None:
            starting_SOC = root_scalar(
                lambda s: negative_SOC_from_cell_SOC(s)
                - initial_socs["Initial concentration in negative electrode"],
                method='toms748',
                bracket=[0, 1],
                x0=0.5
            ).root
        else:
            starting_SOC = None
        data.voltages, returned_SOCs = subtract_both_OCV_curves_from_cycles(
            data,
            parameters,
            negative_SOC_from_cell_SOC,
            positive_SOC_from_cell_SOC,
            starting_SOC=starting_SOC,
        )

    # Apply optional OCV mismatch correction
    if adjust_for_ocv_mismatch:
        ocv_file_index = ocv_data['indices'].index(data.indices[0] - 1)
        ocv_mismatch_start = ocv_mismatch(
            returned_SOCs[0][0],
            ocv_data['OCV [V]'][ocv_file_index],
            parameters,
            electrode,
            voltage_sign
        )
        for i, (pulse_index, relaxation_index) in enumerate(zip(
            data.indices[:-1:2], data.indices[1::2]
        )):
            ocv_file_index = ocv_data['indices'].index(relaxation_index)
            ocv_mismatch_end = ocv_mismatch(
                returned_SOCs[2 * i + 1][-1],
                ocv_data['OCV [V]'][ocv_file_index],
                parameters,
                electrode,
                voltage_sign
            )
            t_pulse = transform_to_unity_interval(data.timepoints[2 * i])
            data.voltages[2 * i] = [
                entry - ((1 - t) * ocv_mismatch_start + t * ocv_mismatch_end)
                for entry, t in zip(data.voltages[2 * i], t_pulse)
            ]
            data.voltages[2 * i + 1] = [
                entry - ocv_mismatch_end for entry in data.voltages[2 * i + 1]
            ]
            ocv_mismatch_start = ocv_mismatch_end

    # Plot
    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4), constrained_layout=True)
    texts = plot_measurement(fig, ax, data, title, plot_current=False)
    ax.set_ylabel("Overpotential  /  V")
    for text in texts:
        text.set_visible(False)
    outname = file_prefix + '_ocv_alignment.' + format
    if overwrite or not isfile(outname):
        fig.savefig(outname, bbox_inches='tight', pad_inches=0.0)
    for text in texts:
        text.set_visible(True)
    if display:
        plt.show()

    # Print processed Cycling_Information JSON to stdout
    print(data.to_json())


if __name__ == '__main__':
    extract_overpotential()
