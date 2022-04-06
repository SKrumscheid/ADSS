"""This module contains the Adaptive Stratification algorithm."""

from __future__ import annotations
from typing import Callable, Tuple, List

from itertools import combinations, permutations
from math import factorial
import numpy as np
from scipy.special import comb

from .stratum import Hyperrect, Simplex, cartesian_to_barycentric_simplex
from .stratum import Stratum

import logging

np.seterr('raise')


def kuhn_tessellation(N_dim, N_strat) -> Tuple[np.ndarray, np.ndarray]:
    """Produce Kuhn partition of unit hypercube in N_dim.

    For details, see Cuvelier & Scarella, Vectorized algorithms
    for regular tessellations of d-orthotopes and their faces.

    This algorithm returns the tessellation
    along the "main-diagonal" of (0,...,0) to (1,...,1).

    Args:
        N_dim: number of stochastic dimensions
        N_strat: number of strata

    Returns:
        A tuple containing the coordinates of vertices
        (corners of unit hypercube) and the connectivity matrix
        ((0,...,0) and (1,...,1) joint edge of all simplices)
    """
    Y_vertices = [0, 1]
    Y_vertices = [Y_vertices] * N_dim
    Y_vertices = np.meshgrid(*Y_vertices)
    Y_vertices = np.array(Y_vertices).reshape(N_dim, -1)

    ref_vertices = np.triu(np.ones((N_dim, N_dim + 1)), 1)

    permus = np.transpose(np.array(list(permutations(range(N_dim)))))
    me = np.zeros((N_dim + 1, N_strat), dtype="int")
    a = 2 ** np.arange(N_dim)
    for k in range(N_strat):
        for j in range(N_dim + 1):
            me[j, k] = np.inner(a, ref_vertices[permus[:, k], j])

        Y_mod = np.vstack((Y_vertices[:, me[:, k]],
                           np.ones((1, N_dim + 1))))
        if np.linalg.det(Y_mod) < 0:
            temp = me[0, k]
            me[0, k] = me[N_dim, k]
            me[N_dim, k] = temp

    return (Y_vertices, me)


class AdaptiveStratification:
    """Class for Adaptive Stratification Algorithm.

    This is class manages all the attributes and methods to manipulate a list
    of strata, that are necessary for adaptive stratification.
    """

    def __init__(self, f: Callable[[np.ndarray[float]], np.ndarray],
                 N_dim: int, N_max: int, N_new_per_stratum: int, alpha: float,
                 *, N_min: int = 10, type_strat: str, dynamic: bool = False,
                 rand_gen: np.random.Generator = None) -> None:
        """Initialize an instance of Stratum.

        Args:
            f: function handle to quantity of interest.
            N_dim: number of stochastic variables, transformed to unit
                   uniforms.
            N_max: total number of samples to be distributed.
            N_new_per_stratum: average number of samples to be added to each
                               stratum in every iteration.
            N_min: number of samples each strata will at least get per iteration
            alpha: proportion of samples to be distributed optimally.
            type_strat: decides the shape of the stratum to use.
            rand_gen: the random number generator which will be used for
                      debugging purposes. This might be removed in
                      future versions.
        """
        self.logger = logging.getLogger(__name__)

        self.f = lambda x: np.asarray(f(x))  # ensure result is a ndarray
        self.N_dim = N_dim
        self.N_max = N_max
        self.N_new_per_stratum = N_new_per_stratum
        self.alpha = alpha
        self.type_strat = type_strat

        self.dynamic = dynamic
        if dynamic:
            self.alpha_seq = [self.alpha]

        self.all_strata = []

        self.N_min = N_min

        self.rg = np.random.default_rng() if rand_gen is None else rand_gen

        if type_strat == 'hyperrect':
            self._init_hyperrect()
            self.solver = self._hyperrect_iteration
        elif type_strat == 'simplex':
            self._init_simplex()
            self.solver = self._simplex_iteration
        else:
            raise ValueError('Only "hyperrect" or "simplex" '
                             'are supported as stratification type.')

    def _init_hyperrect(self) -> None:
        """Initialize the stratification for a hyperrectangle."""
        y_samples = self.rg.uniform(0, 1, (self.N_new_per_stratum, self.N_dim))
        f_samples = self.f(y_samples)

        if self.N_new_per_stratum > self.N_max:
            raise ValueError('Invalid configuration. '
                             'N_max should be larger than N_new_per_stratum')

        Stratum1 = Hyperrect(1, self.N_dim, np.zeros(self.N_dim),
                             np.ones(self.N_dim), self.N_new_per_stratum,
                             y_samples, f_samples, rand_gen=self.rg)
        Stratum1.create_possible_splits()

        self.all_strata.append(Stratum1)
        self.N_strat = 1

    def _init_simplex(self) -> None:
        """Initialize the stratification for a simplex."""
        self.N_strat = factorial(self.N_dim)
        N_suggest = self.N_new_per_stratum * self.N_strat
        N_new_samp = min(N_suggest, self.N_max)
        y_samples = self.rg.uniform(0, 1, (N_new_samp, self.N_dim))
        f_samples = self.f(y_samples)

        ref_vertices, me = kuhn_tessellation(self.N_dim, self.N_strat)
        self.SIMP = np.transpose(me)

        stratum_tentative, sigma_vec, rot_vertices_temp = self._calc_possible_initialization(ref_vertices, me, y_samples, f_samples)

        # A split is needed to get valid simplices from a hypercube (except 1d)
        N_tot = N_new_samp
        its = 1
        while not np.any(sigma_vec):
            if N_tot >= self.N_max:
                self.logger.info('Reached N_max without finding any variance.')
                raise Exception('Reached N_max without finding any variance.')
            N_new_samp = min(N_suggest, self.N_max - its * N_suggest)
            samples_new = self.rg.uniform(0, 1, (N_new_samp, self.N_dim))
            f_samples_new = self.f(samples_new)
            y_samples = np.append(y_samples, samples_new, axis=0)
            f_samples = np.append(f_samples, f_samples_new, axis=0)
            stratum_tentative, sigma_vec, rot_vertices_temp = self._calc_possible_initialization(ref_vertices, me, y_samples, f_samples)
            N_tot += N_new_samp
            its += 1

        # Choose the tesselation with the least variance
        min_var_rot = np.argmin(np.sum(sigma_vec, axis=0))
        self.Y_vertices = rot_vertices_temp[min_var_rot].T

        self.all_strata = stratum_tentative[min_var_rot]

        for strat in self.all_strata:
            strat.create_possible_splits()

    def _calc_possible_initialization(
            self, ref_vertices: np.ndarray, me: np.ndarray,
            y_samples: np.ndarray, f_samples: np.ndarray
            ) -> Tuple[List[Simplex], List[float], List[np.ndarray]]:
        """Return all possible initial Stratification with simplices.

        Args:
            ref_vertices: Array containing every vertices in the stratification.
            me: Array containing the simplices in the stratification.
            y_samples: The initial sample points to determine the optimal stratification.
            f_samples: The initial evaluated sample points to determine the optimal stratification.

        Returns:
            A tuple containing all possible initial tessellations, their
            variance, and their vertices.
        """
        # We try all possible tessellation
        stratum_tentative = [[None for x in range(self.N_strat)]
                             for x in range(2 ** self.N_dim - 1)]
        sigma_vec = [[None for x in range(2 ** self.N_dim - 1)] for x in range(self.N_strat)]
        its = -1
        rot_vertices_temp = [None] * (2 ** self.N_dim - 1)

        # how many values will be rotated
        for rot_dir in range(self.N_dim):
            dim_comb = np.array(
                list(combinations(range(self.N_dim), rot_dir + 1)))

            # combinations how to rotate
            for k in range(comb(self.N_dim, rot_dir + 1, exact=True)):
                its += 1
                rot_vertices = np.copy(ref_vertices)

                # do the rotation and change the tessellation matrix
                for j in range(rot_dir + 1):
                    rot_vertices[dim_comb[k, j], :] = (
                        1 - rot_vertices[dim_comb[k, j], :])

                # to save all tessellations
                rot_vertices_temp[its] = rot_vertices
                for s in range(self.N_strat):
                    lambda_values = cartesian_to_barycentric_simplex(
                        y_samples, rot_vertices[:, me[:, s]])

                    simplex_ind = np.nonzero(
                        np.sum(lambda_values, axis=1)
                        * np.prod(lambda_values >= 0, axis=1)
                        * np.prod(lambda_values <= 1, axis=1))[0]
                    if simplex_ind.size < 1:
                        raise Exception('Empty simplex. '
                                        f'N_max set to {self.N_max} for '
                                        f'{self.N_strat} initial strata.')
                    else:
                        local_y_samples = y_samples[simplex_ind]
                        local_f_samples = f_samples[simplex_ind]

                        # determine corresponding vertices and simplicies
                        vertex_coordinates = rot_vertices.T[
                            self.SIMP[s, 0], :].reshape(-1, 1)
                        for m in range(1, self.N_dim + 1):
                            vertex_coordinates = np.hstack((
                                vertex_coordinates,
                                rot_vertices.T[
                                    self.SIMP[s, m], :].reshape(-1, 1)))

                        temp_strat = Simplex(
                            1 / self.N_strat, self.N_dim,
                            local_f_samples.shape[0], s, vertex_coordinates,
                            local_y_samples, local_f_samples, rand_gen=self.rg)

                        stratum_tentative[its][s] = temp_strat

                        sigma_vec[s][its] = temp_strat.sigma

        return (stratum_tentative, sigma_vec, rot_vertices_temp)

    def _optimal_weights(self) -> Tuple[List[float], int, int]:
        """Return the optimal weights.

        Returns:
            A tuple which contains the optimal weight distribution for the new
            samples, the number of new samples and the new total number of
            samples if we allocate all samples.
        """
        # if sigma_all is a zero vector, proportional allocation will be use
        if any(self.sigma_all):
            self.logger.info('sigma_all is a zero vector, in _optimal_weights')

            # Update optimal allocation
            p_sigma_dot = sum([x * y for x, y in zip(self.p_all, self.sigma_all)])
            q1 = [x * y / p_sigma_dot for x, y in zip(self.p_all, self.sigma_all)]

            # Update proportional allocation
            q2 = self.p_all

            # Compute hybrid allocation
            q = [self.alpha * x + (1 - self.alpha) * y for x, y in zip(q1, q2)]
        else:
            q = self.p_all

        N_new = self.N_strat * self.N_new_per_stratum

        # Check that limit N_max is not exceeded
        if self.N_tot + N_new > self.N_max:
            N_new = self.N_max - self.N_tot

        # The new total (will usually not be satisfied exactly)
        N_tot_aim = self.N_tot + N_new

        return (q, N_new, N_tot_aim)

    def _optimal_allocation(self, q: List[float], N_new: int, N_tot_aim: int,
                            strat_curr: Stratum, i: int) -> int:
        """Return the number of new samples for stratum i based on the weights.

        Uses allocation rule from Etoré and Jourdain (optimal allocation)

        Args:
            q: Optimal weight distribution for the new samples.
            N_new: The number of new samples.
            N_tot_aim: The new total number of samples.
            strat_curr: The current stratum.
            i: The index of the current stratum

        Returns:
            The optimal amount of new samples for the given strata.
        """
        return 1 + max(0, min(round((N_tot_aim - self.N_strat) * q[i] - strat_curr.N_samples), N_new - self.N_strat))

    def solve(self) -> Tuple[float, list[Stratum], int, float]:
        """Run the stratification algorithm.

        It will call the correct solver depending on the specified type.

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

            (split_bool_all, max_var_red_all, max_red_index) = self._update(its)

        # If self.N_tot is managed correctly, they should be the same
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

        return (QoI, self.all_strata, self.N_strat, QoI_var)

    def _hyperrect_iteration(self, to_split: int, split_dim: int) -> None:
        """Run the algorithm for hyperrectangles.

        Args:
            to_split: which stratum should be split
            split_dim: in which dimension the stratum should be split
        """
        if to_split is not False:
            strat1, strat2 = self.all_strata[to_split].split_stratum(split_dim)

            self.all_strata[to_split] = strat1
            self.all_strata.append(strat2)
            self.p_all[to_split] = strat1.p
            self.p_all.append(strat2.p)
            self.sigma_all[to_split] = strat1.sigma
            self.sigma_all.append(strat2.sigma)
            self.N_strat += 1

        q, N_new, N_tot_aim = self._optimal_weights()

        for i in range(self.N_strat):
            # Use allocation rule from Etoré and Jourdain (optimal allocation)
            strat_curr = self.all_strata[i]

            N_new_current_stratum = self._optimal_allocation(q, N_new, N_tot_aim, strat_curr, i)

            if self.N_tot + N_new_current_stratum >= self.N_max:
                N_new_current_stratum = self.N_max - self.N_tot

            # Sample N_new_current_stratum samples and add them.
            y_samp = strat_curr.rand_vectors_on_hyperrectangle(N_new_current_stratum)
            f_samp = self.f(y_samp)

            # New function to update strata with bypass formula (samples and splittings)
            strat_curr.add_samples_both(y_samp, f_samp)

            self.N_tot += N_new_current_stratum

            self.sigma_all[i] = self.all_strata[i].sigma

        # tot_samp only computed for debugging
        tot_samp = 0
        for strat in self.all_strata:
            tot_samp += strat.N_samples

        if tot_samp != self.N_tot:
            print(tot_samp)
            print(self.N_tot)

    def _simplex_iteration(self, to_split: int, split_dim: int) -> None:
        """Run the algorithm for simplices.

        Args:
            to_split: which stratum should be split
            split_dim: in which dimension the stratum should be split
        """
        if to_split is not False:
            self._split_simplex(to_split, split_dim, self.p_all, self.sigma_all)

        q, N_new, N_tot_aim = self._optimal_weights()

        k = self.N_strat
        while k > 0 and self.N_tot < self.N_max:
            k -= 1

            strat_curr = self.all_strata[k]
            N_new_current_stratum = self._optimal_allocation(q, N_new, N_tot_aim, strat_curr, k)

            # Sample N_new_current_stratum samples and add them.
            strat_curr.simplices = k

            if self.N_tot + N_new_current_stratum >= self.N_max:
                N_new_current_stratum = self.N_max - self.N_tot

            if N_new_current_stratum > 0:
                self.N_tot += N_new_current_stratum

                X = strat_curr.rand_stratum(N_new_current_stratum)

                y_samp = X.T
                f_new = self.f(y_samp)

                # Update strata with bypass formula (samples and splittings)
                strat_curr.add_samples_both(y_samp, f_new)

            # Update p and sigma
            self.p_all[k] = strat_curr.p
            self.sigma_all[k] = strat_curr.sigma

        # tot_samp only computed for debugging
        tot_samp = 0
        for strat in self.all_strata:
            tot_samp += strat.N_samples

        if tot_samp != self.N_tot:
            print(tot_samp)
            print(self.N_tot)

    def _hybrid_var_red(self) -> Tuple[list[bool], list[float], list[int]]:
        """Find information about which stratum to split.

        It calculates the variance reduction for each possible split.

        Returns:
            A tuple containing a list of booleans to indicate which strata
            could be split, how high the corresponding reduction would be,
            and which of the splits will give this highest reduction.
        """
        split_bool = []
        var_red = []
        red_in = []

        p = np.array(self.p_all)
        sigma = np.array(self.sigma_all)

        if not np.any(sigma):
            self.logger.info('No variance detected in the stratification'
                             ' using %d samples currently.', self.N_tot)
            return ([False], [0], [None])

        for i, strat_curr in enumerate(self.all_strata):
            if strat_curr.sigma == 0:
                red_in.append(None)
                var_red.append(0)
                split_bool.append(False)
                continue

            ind_loo = np.delete(np.arange(self.N_strat), i)
            p_split = np.hstack((p[ind_loo], 0.5 * p[i], 0.5 * p[i]))
            var_red_temp = []
            if self.type_strat == 'hyperrect':
                sigma_tilde = strat_curr.sigma ** 2 / (
                    1 - self.alpha + self.alpha * strat_curr.sigma
                    / np.inner(p, sigma))
                for j in range(self.N_dim):
                    split1, split2 = strat_curr.possible_splits[j]

                    sigma_split = np.hstack((sigma[ind_loo],
                                             split1.sigma, split2.sigma))
                    is_zero_variance = not np.any(sigma_split)
                    if is_zero_variance:
                        self.logger.info('No variance detected after splitting'
                                         ' stratum %d using %d samples and %d '
                                         'strata.', i, self.N_tot, self.N_strat)
                        var_red_temp.append(strat_curr.p * sigma_tilde)
                    else:
                        if self.N_strat > 1:
                            sig_p_split = np.inner(p_split, sigma_split)
                            sigma_tilde_split1 = split1.sigma ** 2 / (
                                1 - self.alpha
                                + self.alpha * split1.sigma / sig_p_split)
                            sigma_tilde_split2 = split2.sigma ** 2 / (
                                1 - self.alpha
                                + self.alpha * split2.sigma / sig_p_split)
                        else:
                            p_sig_one = p * (split1.sigma + split2.sigma)
                            sigma_tilde_split1 = split1.sigma ** 2 / (
                                1 - self.alpha
                                + 2 * self.alpha * split1.sigma / p_sig_one)
                            sigma_tilde_split2 = split2.sigma ** 2 / (
                                1 - self.alpha
                                + 2 * self.alpha * split2.sigma / p_sig_one)

                        var_red_temp.append(
                            strat_curr.p * sigma_tilde
                            - split1.p * sigma_tilde_split1
                            - split1.p * sigma_tilde_split2)

            elif self.type_strat == 'simplex':
                for j in range(comb(self.N_dim + 1, 2, exact=True)):
                    split1, split2 = strat_curr.possible_splits[j][0:2]

                    sigma_split = np.hstack((sigma[ind_loo],
                                             split1.sigma, split2.sigma))
                    is_zero_vector = not np.any(sigma_split)
                    if is_zero_vector:
                        self.logger.info('No variance detected after splitting'
                                         ' stratum %d using %d samples and %d '
                                         'strata.', i, self.N_tot, self.N_strat)
                    if np.any(sigma) and not is_zero_vector:
                        devi_old = self.alpha * sigma / np.inner(p, sigma)
                        devi_new = self.alpha * sigma_split / np.inner(p_split, sigma_split)
                        term1 = (sigma ** 2 / (1 - self.alpha + devi_old))
                        term2 = (sigma_split ** 2 / (1 - self.alpha + devi_new))
                        reduc = np.inner(p, term1) - np.inner(p_split, term2)
                        var_red_temp.append(reduc)
                    else:
                        var_red_temp.append(0)

            if np.sum(var_red_temp) > 0:
                max_red_index = np.argmax(var_red_temp)
                red_in.append(max_red_index)
                var_red.append(var_red_temp[max_red_index])
                split_bool.append(True)
            else:
                red_in.append(None)
                var_red.append(0)
                split_bool.append(False)

        return (split_bool, var_red, red_in)

    def _split_simplex(self, to_split: int, split_dim: int,
                       p_all: list[float], sigma_all: list[float]) -> None:
        """Split a simplex at split_dim.

        Splits simplex while updating global list.

        Args:
            to_split: which stratum should be split
            split_dim: in which dimension the stratum should be split
            p_all: all probabilities
            sigma_all: all variances
        """
        strat_split = self.all_strata[to_split]
        (strat1, strat2,
         vertex_new, vertex_comb) = strat_split.split_stratum(split_dim)

        if strat_split.N_samples != strat1.N_samples + strat2.N_samples:
            print('Simplex split')
            print(strat_split.N_samples
                  - (strat1.N_samples + strat2.N_samples))

        index_new_vertex = self.Y_vertices.shape[0]
        split_global_vertex_index = self.SIMP[strat_split.simplices]
        split_shared_vertex_indices = np.setdiff1d(
            np.array(range(self.N_dim + 1)), vertex_comb)

        simp_new1 = np.hstack((index_new_vertex,
                              split_global_vertex_index[vertex_comb[0]],
                              split_global_vertex_index[
                                  split_shared_vertex_indices]))
        simp_new2 = np.hstack((index_new_vertex,
                              split_global_vertex_index[vertex_comb[1]],
                              split_global_vertex_index[
                                  split_shared_vertex_indices]))

        self.SIMP[to_split] = simp_new1
        self.SIMP = np.append(self.SIMP, [simp_new2], axis=0)

        self.Y_vertices = np.vstack((self.Y_vertices,  vertex_new.T))

        # Replace the old stratum with one of the two new strata
        # and append the other like in hyperrect
        self.all_strata[to_split] = strat1
        self.all_strata.append(strat2)

        strat1.simplices = to_split
        strat2.simplices = self.N_strat

        p_all[to_split] = strat1.p
        p_all.append(strat2.p)

        sigma_all[to_split] = strat1.sigma
        sigma_all.append(strat2.sigma)

        self.N_strat += 1

    def _asymptotic_var(self, alvec: np.nadarray, sivec: List[float], pvec: List[float],
                        kapvec: np.nadarray) -> Tuple[np.nadarray, np.nadarray]:
        """This is an implementation of variance constant.

        The variance constant of the stratification estimator
        and its asymptotic (CLT) variance are used in sect. 4.4.

        convention: Var(esitimator) = R/N
        R is a function of alpha as well as the vectors sigma, kappa, and p
        its gradient with respect to sigma is dR

        Args:
            alvec: all candidates
            sivec: all variances
            pvec: all probabilities
            kapvec: all kurtosis

        Returns:
          The asymptotic variance of R estimator and the
          variance reduction factor R for each value of alpha in alvec.
        """
        # Transform to numpy arrays
        pvec = np.array(pvec)
        sivec = np.array(sivec)

        avarvec = list(np.zeros(alvec.shape))
        Rvec = np.zeros(alvec.shape)

        psiprod = np.dot(pvec, sivec)

        R = lambda alpha: np.sum(pvec * (sivec ** 2) / (1 + alpha * (sivec / psiprod - 1)))

        dR = lambda alpha: (psiprod * pvec * sivec / (
            alpha * sivec + (1 - alpha) * psiprod)) * (1 + (1 - alpha) * psiprod / (
                alpha * sivec + (1 - alpha) * psiprod)) + alpha * pvec * np.sum(
                    pvec * (sivec ** 3) / ((alpha * sivec + (1 - alpha) * psiprod) ** 2))
        S = lambda alpha: np.diag(psiprod * (sivec ** 2) * (kapvec - 1) / (4 * pvec * ((1 - alpha) * psiprod + alpha * sivec))) 
        asymvar = lambda alpha: np.dot(dR(alpha), S(alpha) @ dR(alpha))

        for i in range(alvec.size):
            aa = alvec[i]
            Rvec[i] = R(aa)
            avarvec[i] = asymvar(aa)

        return (np.array(avarvec), Rvec)

    def _update(self, its: int) -> float:
        """Update alpha and calls _hybrid_var_red.

        Args:
            its: current iteration

        Notes:
            See section 4.4 for a detailed explanation
        """
        if self.dynamic is True:
            # Update p and sigma and kappa
            NN_curr = 0
            kappa = np.empty(self.N_strat)
            for ns in range(self.N_strat):
                strat_curr = self.all_strata[ns]

                self.p_all[ns] = strat_curr.p
                self.sigma_all[ns] = strat_curr.sigma

                X = strat_curr.f_samples
                NN_curr = NN_curr + X.size
                kdebw = np.max([1e-3, 1.06 * (strat_curr.sigma) * X.size ** (-1/5)])  # max (small value, ...) needed, since var = 0 zeros would give kdebw=0
                kdemean = np.mean(X)
                kdem2 = np.mean((X-kdemean) ** 2 + kdebw ** 2)
                kdem4 = np.mean((X-kdemean) ** 4 + 6 * kdebw ** 2 * (X-kdemean) ** 2 + 3 * kdebw ** 4)
                kappa[ns] = kdem4/kdem2 ** 2

            (split_bool_all, max_var_red_all, max_red_index) = self._hybrid_var_red()

            # only update alpha if all variances are not zero
            if(np.linalg.norm(self.sigma_all) > np.finfo(float).eps):
                alist = np.arange(0, 0.975, 0.025)
                (avar, Rval) = self._asymptotic_var(alist, self.sigma_all, self.p_all, kappa)
                ial = np.nonzero(np.abs(alist - self.alpha) < np.finfo(float).eps)[0][0]
                cCI = 1
                impro = Rval[ial] + cCI * np.sqrt(avar[ial]) / np.sqrt(NN_curr) - (Rval + cCI * np.sqrt(avar) / np.sqrt(NN_curr))
                optimpro = np.max(impro)
                fac = 0.8
                cand = np.nonzero(impro > fac * optimpro)[0]

                if(cand.size > 0):
                    ial = np.min(cand)

                # update alpha value
                self.alpha = alist[ial]
                self.alpha_seq.append(self.alpha)
            else:
                self.alpha_seq.append(self.alpha_seq[its])
        else:
            (split_bool_all, max_var_red_all, max_red_index) = self._hybrid_var_red()

        return (split_bool_all, max_var_red_all, max_red_index)
