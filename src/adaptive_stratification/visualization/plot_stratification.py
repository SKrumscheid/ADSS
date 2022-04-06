"""This module adds plotting capabilities to the stratification algorithm."""

from __future__ import annotations
from typing import Callable, Tuple

import numpy as np
from matplotlib.patches import Rectangle  # , Wedge
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt  # eventplot


from ..stratification import AdaptiveStratification
from ..stratum import Stratum


def plot_1D_stratum_hyper(stratification: AdaptiveStratification,
                          scatter_size: float = 1, line_width: float = 1,
                          *, show_samples: bool = True) -> None:
    """Visualize a 1D stratification with hyperrects.

    Args:
        scatter_size: The size of the samples.
        line_width: The edge width of the rectangles.
        show_samples: Whether to show the samples.
    """
    assert(stratification.N_dim == 1)

    patches = []

    fig, ax = plt.subplots()

    for strat in stratification.all_strata:
        patches.append(Rectangle((strat.lower_bounds, -0.05),
                                 (strat.upper_bounds
                                  - strat.lower_bounds), 0.1))
        if show_samples is True:
            # ax.bar((strat.upper_bounds + strat.lower_bounds) / 2,
            #         strat.N_samples, strat.upper_bounds - strat.lower_bounds)
            # n, bins, patches = plt.hist(x, 50, density=True,
            #                             facecolor='g', alpha=0.75)
            ax.scatter(strat.samples, np.zeros(strat.N_samples), s=scatter_size)

    # It's 'None' (with quotes) and not None
    collection = PatchCollection(patches, alpha=0.7, edgecolor='r',
                                 facecolor='None', lw=line_width)
    ax.add_collection(collection)
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    plt.show()


def plot_2D_stratum_hyper(stratification: AdaptiveStratification,
                          scatter_size: float = 1, line_width: float = 1,
                          *, show_samples: bool = True) -> None:
    """Visualize a 2D stratification with hyperrects.

    Args:
        scatter_size: The size of the samples.
        line_width: The edge width of the rectangles.
        show_samples: Whether to show the samples.
    """
    assert(stratification.N_dim == 2)

    patches = []

    fig, ax = plt.subplots()

    for strat in stratification.all_strata:
        patches.append(Rectangle(strat.lower_bounds,
                                 *(strat.upper_bounds
                                   - strat.lower_bounds)))
        if show_samples is True:
            samp = strat.samples
            ax.scatter(samp[:, 0], samp[:, 1], s=scatter_size)

    # Specific to testfun case 1
    # patches.append(Wedge((0, 0), 0.7978845608028654, 0, 90))

    # It's 'None' (with quotes) and not None
    collection = PatchCollection(patches, alpha=0.7, edgecolor='r',
                                 facecolor='None', lw=line_width)
    ax.add_collection(collection)

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    plt.show()


def plot_1D_stratum_simp(stratification: AdaptiveStratification,
                         scatter_size: float = 1, line_width: float = 1,
                         *, show_samples: bool = True) -> None:
    """Visualize a 1D stratification with simplices.

    Args:
       scatter_size: The size of the samples.
       line_width: The edge width of the rectangles.
       show_samples: Whether to show the samples.
    """
    assert(stratification.N_dim == 1)

    patches = []

    fig, ax = plt.subplots()

    for strat in stratification.all_strata:
        bounds = np.sort(strat.vertices[0])
        patches.append(Rectangle((bounds[0], -0.05), bounds[1] - bounds[0], 0.1))
        if show_samples is True:
            ax.scatter(strat.samples, np.zeros(strat.N_samples), s=scatter_size)

    # It's 'None' (with quotes) and not None
    collection = PatchCollection(patches, alpha=0.7, edgecolor='r',
                                 facecolor='None', lw=line_width)
    ax.add_collection(collection)

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    plt.show()


def plot_2D_stratum_simp(stratification: AdaptiveStratification,
                         scatter_size: float = 1, line_width: float = 1,
                         *, show_samples: bool = True) -> None:
    """Visualize a 2D stratification with simplices.

    Args:
       scatter_size: The size of the samples.
       line_width: The edge width of the rectangles.
       show_samples: Whether to show the samples.
    """
    assert(stratification.N_dim == 2)

    fig, ax = plt.subplots()

    ax.triplot(stratification.Y_vertices.T[0], stratification.Y_vertices.T[1],
               stratification.SIMP)

    if show_samples is True:
        for strat in stratification.all_strata:
            samp = strat.samples
            ax.scatter(samp[:, 0], samp[:, 1], s=scatter_size)

    # specific to testfun case 1
    # ax.add_patch(Wedge((0, 0), 0.7978845608028654, 0, 90,
    #                   ec='r', fc='None', lw=1))

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    plt.show()


class AdaptiveStratificationVisualization(AdaptiveStratification):
    def __init__(self, f: Callable[[np.ndarray[float]], np.ndarray],
                 N_dim: int, N_max: int, N_new_per_stratum: int,
                 alpha: float, *, type_strat: str, dynamic: bool = False,
                 rand_gen: np.random.Generator = None) -> None:
        super().__init__(f, N_dim, N_max, N_new_per_stratum, alpha,
                         type_strat=type_strat, dynamic=dynamic,
                         rand_gen=rand_gen)

    def plotter(self) -> None:
        """Plot the current stratification."""
        if self.N_dim == 1:
            if self.type_strat == 'hyperrect':
                plot_1D_stratum_hyper(self)
            elif self.type_strat == 'simplex':
                plot_1D_stratum_simp(self)
            else:
                raise ValueError('Unknown stratification type')
        elif self.N_dim == 2:
            if self.type_strat == 'hyperrect':
                plot_2D_stratum_hyper(self)
            elif self.type_strat == 'simplex':
                plot_2D_stratum_simp(self)
            else:
                raise ValueError('Unknown stratification type')
        else:
            raise ValueError('Not implemented')

    # reusing code from stratification.py
    def solve_vis_result(self) -> Tuple[float, list[Stratum], int, float]:
        """Run the stratification algorithm.

        It will call the correct solver depending on the specified type. It
        will visualize the end result.

        Returns:
            The list containing all strata created in the process, the number
            of strata, the SS estimator of mean and variance of f
        Notes:
            heavy on memory if all strata are stored
        """
        self.p_all = []
        self.sigma_all = []
        self.N_tot = 0

        for strat_curr in self.all_strata:
            self.N_tot += strat_curr.N_samples
            self.p_all.append(strat_curr.p)
            self.sigma_all.append(strat_curr.sigma)

        (split_bool_all, max_var_red_all, max_red_index) = self._hybrid_var_red()

        # Prevent a split for simplices in the first iteration
        if self.type_strat == 'simplex':
            split_bool_all = [False] * len(split_bool_all)
            split_dim = None  # to prevent an error

        # Iterate until N_max is reach
        its = 0  # usage?
        while self.N_tot < self.N_max:
            its += 1

            # Find which stratum and where to split
            if any(split_bool_all):
                # Check why we multiply split_bool_all, give different name?
                max_var_red_all = [x * y for x, y in zip(max_var_red_all,
                                                         split_bool_all)]

                # to_split is which stratum to split
                to_split = np.argmax(max_var_red_all)
                # and split_dim is which dim in that Stratum to split
                split_dim = max_red_index[to_split]
            else:
                to_split = False
                split_dim = None

            self.solver(to_split, split_dim)

            (split_bool_all, max_var_red_all, max_red_index) = self._update(its)

        tot_samp = 0
        for strat in self.all_strata:
            tot_samp += strat.N_samples

        # extra check
        if tot_samp != self.N_max:
            print('Not using N_max samples: ')
            print(tot_samp)

        QoI = 0
        QoI_var = 0
        for i in range(self.N_strat):
            QoI = QoI + self.all_strata[i].p * self.all_strata[i].mean
            QoI_var = (QoI_var + (self.all_strata[i].p) ** 2
                       * (self.all_strata[i].sigma) ** 2
                       / self.all_strata[i].N_samples)

        self.plotter()

        return (QoI, self.all_strata, self.N_strat, QoI_var)

    def solve_vis_steps(self) -> Tuple[float, list[Stratum], int, float]:
        """Run the stratification algorithm.

        It will call the correct solver depending on the specified type. It
        will visualize the result after each step.

        Returns:
            The list containing all strata created in the process, the number
            of strata, the SS estimator of mean and variance of f
        Notes:
            heavy on memory if all strata are stored
        """
        self.p_all = []
        self.sigma_all = []
        self.N_tot = 0

        for strat_curr in self.all_strata:
            self.N_tot += strat_curr.N_samples
            self.p_all.append(strat_curr.p)
            self.sigma_all.append(strat_curr.sigma)

        (split_bool_all, max_var_red_all, max_red_index) = self._hybrid_var_red()

        # Prevent a split for simplices in the first iteration
        if self.type_strat == 'simplex':
            split_bool_all = [False] * len(split_bool_all)
            split_dim = None  # to prevent an error

        # Iterate until N_max is reach
        its = 0
        while self.N_tot < self.N_max:
            its += 1

            # Find which stratum and where to split
            if any(split_bool_all):
                max_var_red_all = [x * y for x, y in zip(max_var_red_all,
                                                         split_bool_all)]

                # to_split is which stratum to split
                to_split = np.argmax(max_var_red_all)
                # and split_dim is which dim in that Stratum to split
                split_dim = max_red_index[to_split]
            else:
                to_split = False
                split_dim = None

            self.solver(to_split, split_dim)

            self.plotter()

            (split_bool_all, max_var_red_all, max_red_index) = self._update(its)

        tot_samp = 0
        for strat in self.all_strata:
            tot_samp += strat.N_samples

        # extra check
        if tot_samp != self.N_max:
            print('Not using N_max samples: ')
            print(tot_samp)

        QoI = 0
        QoI_var = 0
        for i in range(self.N_strat):
            QoI = QoI + self.all_strata[i].p * self.all_strata[i].mean
            QoI_var = (QoI_var + (self.all_strata[i].p) ** 2
                       * (self.all_strata[i].sigma) ** 2
                       / self.all_strata[i].N_samples)

        self.plotter()

        return (QoI, self.all_strata, self.N_strat, QoI_var)
