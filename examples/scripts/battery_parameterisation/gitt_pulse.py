import numpy as np

import pybop

# Define model
parameter_set = pybop.ParameterSet("Xu2019")
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, options={"working electrode": "positive"}
)

# Generate data
sigma = 1e-3
initial_state = {"Initial SoC": 0.9}
experiment = pybop.Experiment(
    [
        "Rest for 1 second",
        "Discharge at 1C for 10 minutes (10 second period)",
        "Rest for 20 minutes",
    ]
)
values = model.predict(initial_state=initial_state, experiment=experiment)
corrupt_values = values["Voltage [V]"].data + np.random.normal(
    0, sigma, len(values["Voltage [V]"].data)
)

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": values["Time [s]"].data,
        "Current function [A]": values["Current [A]"].data,
        "Discharge capacity [A.h]": values["Discharge capacity [A.h]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Define parameter set
parameter_set = pybop.lithium_ion.SPDiffusion.apply_parameter_grouping(
    model.parameter_set, electrode="positive"
)

# Fit the GITT pulse using the single particle diffusion model
gitt_fit = pybop.GITTPulseFit(
    gitt_pulse=dataset,
    parameter_set=parameter_set,
    electrode="positive",
)
gitt_results = gitt_fit()

# Plot the timeseries output
pybop.plot.problem(
    gitt_fit.problem, problem_inputs=gitt_results.x, title="Optimised Comparison"
)

# Plot convergence
pybop.plot.convergence(gitt_fit.optim)

# Plot the parameter traces
pybop.plot.parameters(gitt_fit.optim)
