import numpy as np


class BaseCost:
    """
    Base class for defining cost functions.

    This class is intended to be subclassed to create specific cost functions
    for evaluating model predictions against a set of data. The cost function
    quantifies the goodness-of-fit between the model predictions and the
    observed data, with a lower cost value indicating a better fit.

    Parameters
    ----------
    problem : object
        A problem instance containing the data and functions necessary for
        evaluating the cost function.
    _target : array-like
        An array containing the target data to fit.
    x0 : array-like
        The initial guess for the model parameters.
    bounds : tuple
        The bounds for the model parameters.
    n_parameters : int
        The number of parameters in the model.
    """

    def __init__(self, problem):
        self.problem = problem
        if problem is not None:
            self._target = problem._target
            self.x0 = problem.x0
            self.bounds = problem.bounds
            self.n_parameters = problem.n_parameters

    def __call__(self, x, grad=None):
        """
        Call the evaluate function for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost.
        """
        try:
            return self._evaluate(x, grad)

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def _evaluate(self, x, grad=None):
        """
        Calculate the cost function value for a given set of parameters.

        This method must be implemented by subclasses.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The calculated cost function value.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Call _evaluateS1 for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        try:
            return self._evaluateS1(x)

        except Exception as e:
            raise ValueError(f"Error in cost calculation: {e}")

    def _evaluateS1(self, x):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by the subclass.
        """
        raise NotImplementedError


class RootMeanSquaredError(BaseCost):
    """
    Root mean square error cost function.

    Computes the root mean square error between model predictions and the target
    data, providing a measure of the differences between predicted values and
    observed values.

    Inherits all parameters and attributes from ``BaseCost``.

    """

    def __init__(self, problem):
        super(RootMeanSquaredError, self).__init__(problem)

    def _evaluate(self, x, grad=None):
        """
        Calculate the root mean square error for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The root mean square error.

        """

        prediction = self.problem.evaluate(x)

        if len(prediction) < len(self._target):
            return np.float64(np.inf)  # simulation stopped early
        else:
            return np.sqrt(np.mean((prediction - self._target) ** 2))


class SumSquaredError(BaseCost):
    """
    Sum of squared errors cost function.

    Computes the sum of the squares of the differences between model predictions
    and target data, which serves as a measure of the total error between the
    predicted and observed values.

    Inherits all parameters and attributes from ``BaseCost``.

    Additional Attributes
    ---------------------
    _de : float
        The gradient of the cost function to use if an error occurs during
        evaluation. Defaults to 1.0.

    """

    def __init__(self, problem):
        super(SumSquaredError, self).__init__(problem)

        # Default fail gradient
        self._de = 1.0

    def _evaluate(self, x, grad=None):
        """
        Calculate the sum of squared errors for a given set of parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to evaluate the cost.
        grad : array-like, optional
            An array to store the gradient of the cost function with respect
            to the parameters.

        Returns
        -------
        float
            The sum of squared errors.
        """
        prediction = self.problem.evaluate(x)

        if len(prediction) < len(self._target):
            return np.float64(np.inf)  # simulation stopped early
        else:
            return np.sum(
                (np.sum(((prediction - self._target) ** 2), axis=0)),
                axis=0,
            )

    def _evaluateS1(self, x):
        """
        Compute the cost and its gradient with respect to the parameters.

        Parameters
        ----------
        x : array-like
            The parameters for which to compute the cost and gradient.

        Returns
        -------
        tuple
            A tuple containing the cost and the gradient. The cost is a float,
            and the gradient is an array-like of the same length as `x`.

        Raises
        ------
        ValueError
            If an error occurs during the calculation of the cost or gradient.
        """
        y, dy = self.problem.evaluateS1(x)
        if len(y) < len(self._target):
            e = np.float64(np.inf)
            de = self._de * np.ones(self.problem.n_parameters)
        else:
            dy = dy.reshape(
                (
                    self.problem.n_time_data,
                    self.problem.n_outputs,
                    self.problem.n_parameters,
                )
            )
            r = y - self._target
            e = np.sum(np.sum(r**2, axis=0), axis=0)
            de = 2 * np.sum(np.sum((r.T * dy.T), axis=2), axis=1)

        return e, de

    def set_fail_gradient(self, de):
        """
        Set the fail gradient to a specified value.

        The fail gradient is used if an error occurs during the calculation
        of the gradient. This method allows updating the default gradient value.

        Parameters
        ----------
        de : float
            The new fail gradient value to be used.
        """
        de = float(de)
        self._de = de


class GravimetricEnergyDensity(BaseCost):
    """
    Represents the gravimetric energy density of a battery cell, calculated based
    on a normalized discharge from upper to lower voltage limits. The goal is to
    maximize the energy density, which is achieved by minimizing the negative energy
    density reported by this class.

    Attributes:
        problem (object): The associated problem containing model and evaluation methods.
        parameter_set (object): The set of parameters from the problem's model.
        dt (float): The time step size used in the simulation.
    """

    def __init__(self, problem):
        """
        Initializes the gravimetric energy density calculator with a problem.

        Args:
            problem (object): The problem instance containing the model and data.
        """
        super().__init__(problem)
        self.problem = problem
        self.parameter_set = problem._model._parameter_set
        self.update_simulation_data(problem.x0)

    def update_simulation_data(self, initial_conditions):
        """
        Updates the simulation data based on the initial conditions.

        Args:
            initial_conditions (array): The initial conditions for the simulation.
        """
        self.nominal_capacity(self.problem.x0, self.problem._model)
        solution = self.problem.evaluate(initial_conditions)
        self.problem._time_data = solution[:, -1]
        self.problem._target = solution[:, 0:-1]
        self.dt = solution[1, -1] - solution[0, -1]

    def _evaluate(self, x, grad=None):
        """
        Computes the cost function for the energy density.

        Args:
            x (array): The parameter set for which to compute the cost.
            grad (array, optional): Gradient information, not used in this method.

        Returns:
            float: The negative gravimetric energy density or infinity in case of infeasible parameters.
        """
        try:
            self.nominal_capacity(x, self.problem._model)
            solution = self.problem.evaluate(x)

            voltage, current = solution[:, 0], solution[:, 1]
            negative_energy_density = -np.trapz(voltage * current, dx=self.dt) / (
                3600 * self.cell_mass(self.parameter_set)
            )

            return negative_energy_density

        except UserWarning as e:
            print(f"Ignoring this sample due to: {e}")
            return np.inf

        except Exception as e:
            print(f"An error occurred during the evaluation: {e}")
            return np.inf

    def nominal_capacity(self, x, model):
        """
        Calculate and update the nominal cell capacity based on the theoretical energy density
        and an average voltage.

        The nominal capacity is computed by dividing the theoretical energy (in watt-hours) by
        the average open circuit potential (voltage) of the cell.

        Parameters:
        - x (array-like): An array of values representing the model inputs.
        - model (BatteryModel): An instance of the battery model that contains methods and
                                parameters for calculating the state of health and other
                                properties of the battery.

        Returns:
        - None: The nominal cell capacity is updated directly in the model's parameter set.
        """
        # Extract stoichiometries and compute mean values
        (
            min_sto_neg,
            max_sto_neg,
            min_sto_pos,
            max_sto_pos,
        ) = model._electrode_soh.get_min_max_stoichiometries(model._parameter_set)
        mean_sto_neg = (min_sto_neg + max_sto_neg) / 2
        mean_sto_pos = (min_sto_pos + max_sto_pos) / 2

        inputs = {
            key: x[i]
            for i, key in enumerate([param.name for param in model.parameters])
        }
        model._parameter_set.update(inputs)

        # Calculate theoretical energy density
        theoretical_energy = model._electrode_soh.calculate_theoretical_energy(
            model._parameter_set
        )

        # Calculate average voltage
        positive_electrode_ocp = model._parameter_set["Positive electrode OCP [V]"]
        negative_electrode_ocp = model._parameter_set["Negative electrode OCP [V]"]
        average_voltage = positive_electrode_ocp(mean_sto_pos) - negative_electrode_ocp(
            mean_sto_neg
        )

        # Calculate and update nominal capacity
        theoretical_capacity = theoretical_energy / average_voltage
        model._parameter_set.update(
            {"Nominal cell capacity [A.h]": theoretical_capacity}
        )

    def cell_mass(self, parameter_set):
        """
        Calculate the total cell mass in kilograms.

        This method uses the provided parameter set to calculate the mass of different
        components of the cell, such as electrodes, separator, and current collectors,
        based on their densities, porosities, and thicknesses. It then calculates the
        total mass by summing the mass of each component.

        Parameters:
        - parameter_set (dict): A dictionary containing the parameter values necessary
                                for the mass calculations.

        Returns:
        - float: The total mass of the cell in kilograms.
        """

        def mass_density(
            active_material_vol_frac, density, porosity, electrolyte_density
        ):
            return (active_material_vol_frac * density) + (
                porosity * electrolyte_density
            )

        def area_density(thickness, mass_density):
            return thickness * mass_density

        # Approximations due to SPM(e) parameter set limitations
        electrolyte_density = parameter_set["Separator density [kg.m-3]"]

        # Calculate mass densities
        positive_mass_density = mass_density(
            parameter_set["Positive electrode active material volume fraction"],
            parameter_set["Positive electrode density [kg.m-3]"],
            parameter_set["Positive electrode porosity"],
            electrolyte_density,
        )
        negative_mass_density = mass_density(
            parameter_set["Negative electrode active material volume fraction"],
            parameter_set["Negative electrode density [kg.m-3]"],
            parameter_set["Negative electrode porosity"],
            electrolyte_density,
        )

        # Calculate area densities
        positive_area_density = area_density(
            parameter_set["Positive electrode thickness [m]"], positive_mass_density
        )
        negative_area_density = area_density(
            parameter_set["Negative electrode thickness [m]"], negative_mass_density
        )
        separator_area_density = area_density(
            parameter_set["Separator thickness [m]"],
            parameter_set["Separator porosity"] * electrolyte_density,
        )
        positive_cc_area_density = area_density(
            parameter_set["Positive current collector thickness [m]"],
            parameter_set["Positive current collector density [kg.m-3]"],
        )
        negative_cc_area_density = area_density(
            parameter_set["Negative current collector thickness [m]"],
            parameter_set["Negative current collector density [kg.m-3]"],
        )

        # Calculate cross-sectional area
        cross_sectional_area = (
            parameter_set["Electrode height [m]"] * parameter_set["Electrode width [m]"]
        )

        # Calculate and return total cell mass
        total_area_density = (
            positive_area_density
            + negative_area_density
            + separator_area_density
            + positive_cc_area_density
            + negative_cc_area_density
        )
        return cross_sectional_area * total_area_density
