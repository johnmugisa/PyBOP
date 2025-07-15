from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybamm
from scipy.io import savemat

import pybop

## Group parameter set
R0 = 0.01

parameter_set = pybop.ParameterSet.pybamm("Chen2020")
parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
parameter_set["Electrolyte conductivity [S.m-1]"] = 1e16
parameter_set["Negative electrode conductivity [S.m-1]"] = 1e16
parameter_set["Positive electrode conductivity [S.m-1]"] = 1e16

grouped_parameters = pybop.lithium_ion.GroupedSPMe.apply_parameter_grouping(
    parameter_set
)
grouped_parameters["Series resistance [Ohm]"] = R0

## Create model
model_options = {"surface form": "differential", "contact resistance": "true"}
var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}
model = pybop.lithium_ion.GroupedSPMe(
    parameter_set=grouped_parameters,
    eis=True,
    var_pts=var_pts,
    options=model_options,
    solver=pybamm.CasadiSolver(),
)

## Simulate impedance
Nfreq = 60
fmin = 2e-4
fmax = 1e3
NSOC = 9
frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nfreq)
SOCs = np.linspace(0.1, 0.9, NSOC)

impedances = 1j * np.zeros((Nfreq, NSOC))
for ii, SOC in enumerate(SOCs):
    model.build(initial_state={"Initial SoC": SOC})
    simulation = model.simulateEIS(inputs=None, f_eval=frequencies)
    impedances[:, ii] = simulation["Impedance"]

fig, ax = plt.subplots()
for ii in range(len(SOCs)):
    ax.plot(
        np.real(impedances[:, ii]),
        -np.imag(impedances[:, ii]),
    )
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.grid()
ax.set_aspect("equal", "box")
plt.show()

# Save data
mdic = {"Z": impedances, "f": frequencies, "SOC": SOCs}
current_dir = Path(__file__).parent
save_path = current_dir / "Data" / "Z_SPMegrouped_SOC_chen2020.mat"
savemat(save_path, mdic)
