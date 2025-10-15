# Pybop
# Pybop

# Part one

 This code is a workflow example for pre-processing GITT data (inspired by EP-BOLFI scripts), including:

Step 1: Convert a Parquet file into the pipeline’s JSON format.
Step 2: Automatically fit an exponential decay to pulse/relax segments to generate OCV points vs SOC (via coulomb counting).
Step 3: Fit an interpolating smoothing spline to the OCV points to produce:

(1) An interpolating OCV function (used later as a modeling parameter),
(2) A table of OCV points with corresponding SOC values and indices,
(3) Diagnostic plots of the fit.

Note: Step 3 involves some randomness (e.g., optimizer seeds). You may need to fine-tune input options to achieve a smooth, physically 
consistent fit. See the parameter help in SCRIPTS/fit_and_plot_ocv_local.py --help (or your documentation link).

Step 4: Compute overpotentials by subtracting the static component from the measured voltage. Proper slicing with 
select_measurement_segments is required, and you should update SCRIPTS/overpotential_parameters.py whenever the dataset changes.

Step 5: Convert the overpotential results from JSON to Parquet for efficient storage and analysis.

# Part two

The second part extends PyBoP for non-intrusive parameterization. It fits multiple pulses across the SOC range with per-pulse updates, and 
lets users select which pulses to fit.

 Outputs include:
 • Plots of measured vs fitted voltage for each selected pulse,
 • A CSV file collecting all fitted parameters with their SOC and pulse number for downstream plotting and analysis.
