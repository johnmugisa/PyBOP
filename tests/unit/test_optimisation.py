import numpy as np
import pytest

import pybop


class TestOptimisation:
    """
    A class to test the optimisation class.
    """

    @pytest.fixture
    def dataset(self):
        return pybop.Dataset(
            {
                "Time [s]": np.linspace(0, 360, 10),
                "Current function [A]": np.zeros(10),
                "Terminal voltage [V]": np.ones(10),
            }
        )

    @pytest.fixture
    def parameters(self):
        return [
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Gaussian(0.6, 0.2),
                bounds=[0.58, 0.62],
            )
        ]

    @pytest.fixture
    def problem(self, parameters, dataset):
        model = pybop.lithium_ion.SPM()
        return pybop.FittingProblem(
            model,
            parameters,
            dataset,
            signal=["Terminal voltage [V]"],
        )

    @pytest.fixture
    def cost(self, problem):
        return pybop.SumSquaredError(problem)

    @pytest.mark.parametrize(
        "optimiser_class, expected_name",
        [
            (pybop.SciPyMinimize, "SciPyMinimize"),
            (pybop.SciPyDifferentialEvolution, "SciPyDifferentialEvolution"),
            (pybop.GradientDescent, "Gradient descent"),
            (pybop.Adam, "Adam"),
            (pybop.CMAES, "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"),
            (pybop.SNES, "Seperable Natural Evolution Strategy (SNES)"),
            (pybop.XNES, "Exponential Natural Evolution Strategy (xNES)"),
            (pybop.PSO, "Particle Swarm Optimisation (PSO)"),
            (pybop.IRPropMin, "iRprop-"),
        ],
    )
    @pytest.mark.unit
    def test_optimiser_classes(self, cost, optimiser_class, expected_name):
        opt = pybop.Optimisation(cost=cost, optimiser=optimiser_class)

        assert opt.optimiser is not None
        assert opt.optimiser.name() == expected_name

        # Test without bounds
        cost.bounds = None
        if optimiser_class in [pybop.SciPyDifferentialEvolution]:
            with pytest.raises(ValueError):
                pybop.Optimisation(cost=cost, optimiser=optimiser_class)
        else:
            opt = pybop.Optimisation(cost=cost, optimiser=optimiser_class)

            if optimiser_class in [pybop.SciPyMinimize]:
                assert opt.optimiser.bounds is None
            else:
                assert opt.optimiser.boundaries is None

    @pytest.mark.unit
    def test_default_optimiser(self, cost):
        opt = pybop.Optimisation(cost=cost)
        assert (
            opt.optimiser.name()
            == "Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"
        )

    @pytest.mark.unit
    def test_incorrect_optimiser_class(self, cost):
        class RandomClass:
            pass

        with pytest.raises(ValueError):
            pybop.Optimisation(cost=cost, optimiser=RandomClass)

    @pytest.mark.unit
    def test_prior_sampling(self, cost):
        # Tests prior sampling
        for i in range(50):
            opt = pybop.Optimisation(cost=cost, optimiser=pybop.CMAES)

            assert opt.x0 <= 0.62 and opt.x0 >= 0.58

    @pytest.mark.unit
    def test_halting(self, cost):
        # Test max evalutions
        optim = pybop.Optimisation(cost=cost, optimiser=pybop.GradientDescent)
        optim.set_max_evaluations(1)
        x, __ = optim.run()
        assert optim._iterations == 1

        # Test max unchanged iterations
        optim = pybop.Optimisation(cost=cost, optimiser=pybop.GradientDescent)
        optim.set_max_unchanged_iterations(1)
        optim.set_min_iterations(1)
        x, __ = optim.run()
        assert optim._iterations == 2

        # Test invalid values
        with pytest.raises(ValueError):
            optim.set_max_evaluations(-1)
        with pytest.raises(ValueError):
            optim.set_min_iterations(-1)
        with pytest.raises(ValueError):
            optim.set_max_unchanged_iterations(-1)
        with pytest.raises(ValueError):
            optim.set_max_unchanged_iterations(1, threshold=-1)

    @pytest.mark.unit
    def test_unphysical_result(self, cost):
        # Trigger parameters not physically viable warning
        optim = pybop.Optimisation(cost=cost)
        optim.check_optimal_parameters(np.array([2]))
