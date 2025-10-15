"""
LOCAL-ONLY: Read a Cycling_Information JSON from stdin and extract OCV
points (GITT), writing *_extraction.json and a plot locally.
"""

from ast import literal_eval
import json, sys
import xmlhelpy
import matplotlib.pyplot as plt

@xmlhelpy.command(name="python extract_ocv_curve_local.py", version="${VERSION}")
@xmlhelpy.option("filename", char="n", param_type=xmlhelpy.String, required=True,
                 description="File name template for outputs (no ext needed).")
@xmlhelpy.option("title", char="t", default=None, param_type=xmlhelpy.String,
                 description="Plot title (defaults to template file name).")
@xmlhelpy.option("format", char="f", default="pdf",
                 param_type=xmlhelpy.Choice(
                     ['eps','jpg','jpeg','pdf','pgf','png','ps','raw','rgba','svg','svgz','tif','tiff'],
                     case_sensitive=True),
                 description="Image format.")
@xmlhelpy.option("segments", char="s", default="[(1, None, 2)]", param_type=xmlhelpy.String,
                 description="Segment slice spec as Python list of tuples, e.g. '[(0,2),(3,None,2)]'.")
@xmlhelpy.option("exclude", char="c", default=0.0, param_type=xmlhelpy.Float,
                 description="Seconds to exclude at start of each segment before fitting.")
@xmlhelpy.option("split-on-current-direction", char="l", default=False, param_type=xmlhelpy.Bool,
                 description="If True, write separate lithiation/delithiation extractions.")
@xmlhelpy.option("positive-current-is-lithiation", char="u", default=True, param_type=xmlhelpy.Bool,
                 description="Sign convention for current.")
@xmlhelpy.option("display", char="v", is_flag=True, description="Show the plot interactively.")
def extract_ocv_curve_local(
    filename, title, format, segments, exclude,
    split_on_current_direction, positive_current_is_lithiation, display,
):
    # Imports that depend on ep_bolfi
    from ep_bolfi.utility.dataset_formatting import Cycling_Information
    from ep_bolfi.utility.fitting_functions import fit_exponential_decay
    from ep_bolfi.utility.preprocessing import calculate_SOC, find_occurrences
    from ep_bolfi.utility.visualization import plot_measurement
    from multiprocessing import Pool
    from numpy import mean

    # Load measurement JSON from stdin
    try:
        data = Cycling_Information.from_json(sys.stdin.read())
    except json.decoder.JSONDecodeError:
        raise ValueError("No (or invalid) measurement JSON piped to this tool.")

    seg_spec = literal_eval(segments)
    rest_indices = []
    for seg in seg_spec:
        if len(seg) == 2:
            a, b = seg
            rest_indices.extend(range(len(data.indices) if a is None else a,
                                      len(data.indices) if b is None else b))
        elif len(seg) == 3:
            a, b, c = seg
            rest_indices.extend(range(len(data.indices) if a is None else a,
                                      len(data.indices) if b is None else b, c))
        else:
            raise ValueError("Segments must be 2- or 3-tuples.")

    source_indices = [data.indices[i] for i in rest_indices]
    soc_in_coulomb = calculate_SOC(data.timepoints, data.currents)
    socs = [soc_in_coulomb[i][-1] for i in rest_indices]

    exclude_idx = {
        i: find_occurrences(data.timepoints[i], data.timepoints[i][0] + exclude)[0]
        for i in rest_indices
    }
    parallel_args = [
        (data.timepoints[i][exclude_idx[i]:], data.voltages[i][exclude_idx[i]:])
        for i in rest_indices
    ]
    with Pool() as p:
        fits = p.starmap(fit_exponential_decay, parallel_args)

    ocvs = []
    marker_locations = [(data.timepoints[i][-1] - data.timepoints[0][0]) / 3600
                        for i in rest_indices]

    pop = 0
    for i, r in enumerate(fits):
        if not r:
            # drop segments where fit failed
            j = i - pop
            for arr in (rest_indices, source_indices, socs, marker_locations):
                arr.pop(j)
            pop += 1
        else:
            ocvs.append(r[0][2][0])

    file_prefix = filename.split('.')[0]
    if title is None:
        title = file_prefix

    # Prepare outputs
    if split_on_current_direction:
        pulse_indices = [(ri - 1 if ri > 0 else 0) for ri in rest_indices]
        lith_flip = 1 if positive_current_is_lithiation else -1
        L_socs, L_ocv, L_idx, L_mark = [], [], [], []
        D_socs, D_ocv, D_idx, D_mark = [], [], [], []
        for pi, s, o, si, ml in zip(pulse_indices, socs, ocvs, source_indices, marker_locations):
            if mean(data.currents[pi]) * lith_flip >= 0.0:
                L_socs.append(s); L_ocv.append(o); L_idx.append(si); L_mark.append(ml)
            else:
                D_socs.append(s); D_ocv.append(o); D_idx.append(si); D_mark.append(ml)
        with open(file_prefix + "_lithiation_extraction.json", "w") as f:
            json.dump({"SOC [C]": L_socs, "OCV [V]": L_ocv, "indices": L_idx}, f)
        with open(file_prefix + "_delithiation_extraction.json", "w") as f:
            json.dump({"SOC [C]": D_socs, "OCV [V]": D_ocv, "indices": D_idx}, f)
    else:
        with open(file_prefix + "_extraction.json", "w") as f:
            json.dump({"SOC [C]": socs, "OCV [V]": ocvs, "indices": source_indices}, f)

    # Plot
    fig, ax = plt.subplots(figsize=(4 * 2**0.5, 4), constrained_layout=True)
    texts = plot_measurement(fig, ax, data, title)
    if split_on_current_direction:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.plot(L_mark, L_ocv, marker='1', lw=0, ms=10, color=colors[0], label="lithiation")
        ax.plot(D_mark, D_ocv, marker='1', lw=0, ms=10, color=colors[1], label="delithiation")
        ax.legend()
    else:
        ax.plot(marker_locations, ocvs, marker='1', lw=0, ms=10, color='gray')
    for t in texts: t.set_visible(False)
    fig.savefig(file_prefix + "_extraction." + format, bbox_inches='tight', pad_inches=0.0)
    for t in texts: t.set_visible(True)
    if display: plt.show()

if __name__ == "__main__":
    extract_ocv_curve_local()
