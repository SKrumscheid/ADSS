"""This module includes all classes relevant for Adaptive Stratification."""

from __future__ import annotations
from typing import Tuple
from itertools import combinations
from scipy.special import comb
from math import factorial

import numpy as np


def cartesian_to_barycentric_simplex(x_coords: np.ndarray,
                                     vertex_coords: np.ndarray) -> np.ndarray:
    """Transform the Cartesian coords to the simplex specific Barycentric ones.

    Args:
        vertex_coords: the Cartesian coordinates of the simplex vertices
        x_coords: the samples in Cartesian coordinates

    Returns:
        A NumPy array containing the given values in barycentric coordinates.
    """
    num_pts, dim = x_coords.shape
    T = np.zeros([dim, dim])
    baryc = np.zeros([num_pts, dim + 1])
    T = vertex_coords[:, 0:dim] - vertex_coords[:, dim].reshape(-1, 1)
    for i in range(num_pts):
        baryc[i, 0:dim] = np.linalg.solve(
            T, (x_coords[i, :].T - vertex_coords[:, dim]))
        baryc[i, dim] = 1 - np.sum(baryc[i, 0:dim])

    return baryc[:, 0:dim]


def _rand_simplex(N: int, d: int, C: np.array, z0: np.array) -> np.array:
    """Generate N random vectors in the d-dimensional.

    The generated vectors are spanned by z0, z1, .., zd,
    where columns of C are precisely z1-z0, z2-z0, ... zd-z0.

    Args:
        N: the number of samples to generate
        d: the dimension of the samples
        z0: the first vertex

    Returns:
        A NumPy array containing the random values.

    Notes:
        This is Algorithm 3.24 in "Handbook of Monte Carlo Methods"
        by Kroese et al.
    """
    Y = _rand_unit_simplex_dirichlet(N, d)
    return z0 + C @ Y


def _rand_unit_simplex_dirichlet(N: int, d: int) -> np.array:
    """Generate N random vectors in the d-dimensional unit simplex using Dirichlet sampling.

    Args:
        N: the number of samples to generate
        d: the dimension of the samples

    Returns:
        A NumPy array containing the random values.

    Notes:
        This is Algorithm 3.23 in "Handbook of Monte Carlo Methods" (Kroese et al).
        As this algorithm does not require ot sort numbers, it should be faster
        than the algorithm based on order statistics for large dimensions d.
    """
    X = np.zeros((d, N))
    for n in range(N):
        Y = np.random.default_rng().exponential(scale=1, size=(d+1, 1))
        X[:, [n]] = Y[0:d] / np.sum(Y)

    return X


class Stratum:
    """Class for a stratum.

    This is class holds all the attributes and methods for a stratum for
    an adaptive stratification.
    """

    def __init__(self, p: float, N_dim: int,
                 N_samples: int, samples: np.ndarray, f_samples: np.ndarray, *,
                 rand_gen: np.random.Generator) -> None:
        """Initialize an instance of Stratum.

        Args:
            p: the probability/size of the domain
            N_dim: the Number of Dimensions this stratum resides in
            N_samples: the number of samples in the stratum
            samples: all sample points to be evaluated in the domain
            f_samples: all samples evaluated; an 1xN_samples ndarray
            rand_gen: a fixed random number generator for debugging purposes
        """
        self.p = p
        self.N_dim = N_dim
        self.N_samples = N_samples
        self.samples = samples
        self.f_samples = f_samples
        if N_samples > 1:
            self.mean = np.mean(f_samples)
            self.sigma = np.std(f_samples, ddof=1)
        elif N_samples == 1:
            self.mean = f_samples[0]
            self.sigma = 0
        else:
            self.mean = 0
            self.sigma = 0

        self.possible_splits = []

        self.rg = rand_gen

    def _update_data(self, samples: np.ndarray,
                     f_samples: np.ndarray) -> None:
        """Update samples, f_samples and statistic.

        Args:
            samples: the new samples to be added
            f_samples: the new evaluated samples to be added
        """
        N_old = self.N_samples
        N_add = samples.shape[0]
        if N_add > 1:
            mean_add = np.mean(f_samples)
            sigma_add = np.std(f_samples, ddof=1)
        elif N_add == 1:
            mean_add = np.mean(f_samples)
            sigma_add = 0
        else:
            mean_add = 0
            sigma_add = 0

        self.N_samples += N_add
        self.samples = np.append(self.samples, samples, axis=0)
        self.f_samples = np.append(self.f_samples, f_samples, axis=0)

        if self.N_samples > 1:
            self.sigma = (((N_old - 1) * self.sigma ** 2 + (N_add - 1)
                           * sigma_add ** 2 + N_old * N_add / (self.N_samples)
                           * (self.mean-mean_add)**2) / (self.N_samples - 1))**0.5
            self.mean = ((N_old * self.mean + N_add * mean_add) / self.N_samples)
        elif N_add == 1:
            self.sigma = 0
            self.mean = np.mean(f_samples)
        else:
            self.sigma = 0
            self.mean = 0

    def add_samples_both(self, samples: np.ndarray, f_samples: np.ndarray) -> None:
        """Update the stratum with the newly created samples and f_samples.

        Args:
            samples: new samples points
            f_samples: new samples values
        """
        self._update_data(samples, f_samples)
        self._update_splits(samples, f_samples)


class Hyperrect(Stratum):
    """Class for a hyperrectangle.

    This class holds all attributes and methods specific to a hyperrectangle.
    """

    def __init__(self, p: float, N_dim: int, lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray, N_samples: int, samples: np.ndarray,
                 f_samples: np.ndarray, *,
                 rand_gen: np.random.Generator) -> None:
        """Initialize an instance of Hyperrect.

        Args:
            p: the probability/size of the domain
            N_dim: the Number of Dimensions this stratum resides in
            lower_bounds: array containing the lower bound in each dimension
            upper_bounds: array containing the upper bound in each dimension
            N_samples: the number of samples in the stratum
            samples:aAll sample points to be evaluated in the domain
            f_samples: all samples evaluated
            rand_gen: a fixed random number generator for debugging purposes.
        """
        super().__init__(p, N_dim, N_samples, samples, f_samples,
                         rand_gen=rand_gen)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def rand_vectors_on_hyperrectangle(self, n: int) -> np.ndarray:
        """Create n vectors and return them as a matrix.

        This function creates n row-vectors with values based on
        a uniform distribution over the domain of a hyperrectangle.

        Args:
            n: the number of samples to create

        Returns:
            A matrix containing the n row-vector stacked over each other.
        """
        return (self.lower_bounds + (self.upper_bounds - self.lower_bounds)
                * self.rg.uniform(0, 1, (n, self.N_dim)))

    def create_possible_splits(self) -> None:
        """Create all possible splits as add them to a list.

        There are n possible splits for an n-dimensional hyperrectangle. This
        function creates all the possible splits, while keeping track which of
        the sampling point will end up in which split. It saves this
        information in a list with the name possible_splits.
        """
        if self.possible_splits:
            return

        for i in range(self.N_dim):
            # Split in CDF space:
            split_pt = (self.lower_bounds[i]+self.upper_bounds[i]) / 2

            ind_samples_split1 = np.nonzero(self.samples[:, i] <= split_pt)[0]
            ind_samples_split2 = np.nonzero(self.samples[:, i] > split_pt)[0]

            # --- First substratum --------------------------------------------

            # Copy changed boundary (call by object reference)
            split1_upper = np.copy(self.upper_bounds)
            split1_upper[i] = split_pt

            split1_f_samples = self.f_samples[ind_samples_split1]

            split1 = Hyperrect(self.p / 2, self.N_dim, self.lower_bounds,
                               split1_upper, ind_samples_split1.size,
                               self.samples[ind_samples_split1],
                               split1_f_samples, rand_gen=self.rg)

            # --- Second substratum -------------------------------------------

            split2_lower = np.copy(self.lower_bounds)
            split2_lower[i] = split_pt

            split2_f_samples = self.f_samples[ind_samples_split2]

            split2 = Hyperrect(self.p / 2, self.N_dim, split2_lower,
                               self.upper_bounds, ind_samples_split2.size,
                               self.samples[ind_samples_split2],
                               split2_f_samples, rand_gen=self.rg)

            self.possible_splits.append((split1, split2))

    def split_stratum(self, split_dim: int) -> Tuple[Stratum, Stratum]:
        """Create the two substrata by a split in the desired dimension.

        Args:
            split_dim: the dimension where the split should happen

        Returns:
            The two Strata defined by this split as a tuple.
        """
        split1, split2 = self.possible_splits[split_dim]
        split1.create_possible_splits()
        split2.create_possible_splits()
        return (split1, split2)

    def _update_splits(self, samples: np.ndarray,
                       f_samples: np.ndarray) -> None:
        """Update all_splits with the new samples.

        Args:
            samples: the new samples to be added
            f_samples: the new evaluated samples to be added
        """
        for i in range(self.N_dim):
            split1, split2 = self.possible_splits[i]
            split_pt = (self.lower_bounds[i]+self.upper_bounds[i]) / 2

            ind_samples_split1 = np.nonzero(samples[:, i] <= split_pt)[0]
            ind_samples_split2 = np.nonzero(samples[:, i] > split_pt)[0]

            # --- First substratum --------------------------------------------
            split1_samples = samples[ind_samples_split1]
            split1_f_samples = f_samples[ind_samples_split1]
            split1._update_data(split1_samples, split1_f_samples)
            # --- Second substratum -------------------------------------------
            split2_samples = samples[ind_samples_split2]
            split2_f_samples = f_samples[ind_samples_split2]
            split2._update_data(split2_samples, split2_f_samples)


class Simplex(Stratum):
    """Class for a Simplex.

    This class holds all attributes and methods specific to a simplex.
    """

    def __init__(self, p: float, N_dim: int, N_samples: int,
                 simplices: np.ndarray, vertices: np.ndarray,
                 samples: np.ndarray, f_samples: np.ndarray, *,
                 rand_gen: np.random.Generator) -> None:
        """Initialize an instance of Simplex.

        Args:
            p: the probability/size of the domain.
            N_dim: the Number of Dimensions this stratum resides in.
            N_samples: the number of samples in the stratum
            simplices: the index of the simplex in a global list
            vertices: the Cartesian coordinates of the vertices
            samples: all sample points to be evaluated in the domain
            f_samples: all samples evaluated
            rand_gen: a fixed random number generator for debugging purposes
        """
        super().__init__(p, N_dim, N_samples, samples, f_samples,
                         rand_gen=rand_gen)

        self.simplices = simplices
        self.vertices = vertices

    def create_possible_splits(self) -> None:
        """Create all possible splits as add them to a list.

        There are n*(n-1)/2 possible splits for an n-dimensional simplex. This
        function creates all the possible splits, while keeping track which of
        the sampling point will end up in which split. It saves this
        information in a list with the name possible_splits.
        """
        if self.possible_splits:
            return

        vertex_comb_array = np.array(list(
            combinations(range(self.N_dim + 1), 2)))

        lambda_values = cartesian_to_barycentric_simplex(self.samples, self.vertices)
        lambda_values = np.hstack((lambda_values, 1-np.sum(lambda_values, 1).reshape(-1, 1)))

        for j in range(comb(self.N_dim + 1, 2, exact=True)):
            vertex_new = np.mean(
                self.vertices[:, vertex_comb_array[j, :]],
                axis=1).reshape(-1, 1)
            vertex_comb = vertex_comb_array[j, :]

            # --- First substratum --------------------------------------------

            vertex_temp_coordinates = np.hstack((
                vertex_new,
                self.vertices[:, vertex_comb[0]].reshape(-1, 1),
                self.vertices[:, np.setdiff1d(
                    np.array(range(self.N_dim + 1)), vertex_comb)]))

            if self.N_samples > 0:
                samples_in_s1 = np.nonzero(
                    lambda_values[:, vertex_comb_array[j, 0]]
                    > lambda_values[:, vertex_comb_array[j, 1]])[0]
            else:
                samples_in_s1 = np.array([], dtype=int)

            split1 = Simplex(self._volume_simplex(vertex_temp_coordinates),
                             self.N_dim, samples_in_s1.shape[0], -1,
                             vertex_temp_coordinates,
                             self.samples[samples_in_s1],
                             self.f_samples[samples_in_s1], rand_gen=self.rg)

            # --- Second substratum -------------------------------------------

            vertex_temp_coordinates = np.hstack((
                vertex_new,
                self.vertices[:, vertex_comb[1]].reshape(-1, 1),
                self.vertices[:, np.setdiff1d(
                    np.array(range(self.N_dim + 1)), vertex_comb)]))

            # Find the samples in simplex 2
            if self.N_samples > 0:
                samples_in_s2 = np.arange(self.N_samples)
                samples_in_s2 = np.delete(samples_in_s2, samples_in_s1)
            else:
                samples_in_s2 = np.array([], dtype=int)  # review

            split2 = Simplex(self._volume_simplex(vertex_temp_coordinates),
                             self.N_dim, samples_in_s2.shape[0], -1,
                             vertex_temp_coordinates,
                             self.samples[samples_in_s2],
                             self.f_samples[samples_in_s2], rand_gen=self.rg)

            # Add the split & var_red to lists
            self.possible_splits.append((split1, split2,
                                         vertex_new, vertex_comb))

    def split_stratum(self, split_dim: int) -> Tuple[Stratum, Stratum,
                                                     np.ndarray, np.ndarray]:
        """Create the two substrata by a split in the desired dimension.

        Args:
            split_dim: the dimension where the split should happen

        Returns:
            A tuple containing the two strata defined by this split, the
            new vertex, and the two vertex indices where the new vertex lies
            in between.
        """
        (split1, split2, vertex_new, vertex_comb) = self.possible_splits[split_dim]
        split1.create_possible_splits()
        split2.create_possible_splits()
        return (split1, split2, vertex_new, vertex_comb)

    def _update_splits(self, samples: np.ndarray,
                       f_samples: np.ndarray) -> None:
        """Update all_splits with the new samples.

        Args:
            samples: The new samples to be added.
            f_samples: The new evaluated samples to be added.
        """
        vertex_comb_array = np.array(list(
            combinations(range(self.N_dim + 1), 2)))

        lambda_values = cartesian_to_barycentric_simplex(samples, self.vertices)
        lambda_values = np.hstack((lambda_values, 1-np.sum(lambda_values, 1).reshape(-1, 1)))

        for j in range(comb(self.N_dim + 1, 2, exact=True)):
            split1, split2 = self.possible_splits[j][0:2]

            # --- First substratum --------------------------------------------

            if self.N_samples > 0:
                samples_in_s1 = np.nonzero(
                    lambda_values[:, vertex_comb_array[j, 0]]
                    > lambda_values[:, vertex_comb_array[j, 1]])[0]
            else:
                samples_in_s1 = np.array([], dtype=int)

            split1_samples = samples[samples_in_s1]
            split1_f_samples = f_samples[samples_in_s1]
            split1._update_data(split1_samples, split1_f_samples)

            # --- Second substratum -------------------------------------------

            if self.N_samples > 0:
                samples_in_s2 = np.arange(samples.shape[0])  # review
                samples_in_s2 = np.delete(samples_in_s2, samples_in_s1)
            else:
                samples_in_s2 = np.array([], dtype=int)

            split2_samples = samples[samples_in_s2]
            split2_f_samples = f_samples[samples_in_s2]
            split2._update_data(split2_samples, split2_f_samples)

    def _volume_simplex(self, vertex_coords: np.ndarray) -> float:
        """Return the volume of the simplex.

        Args:
            vertex_coords: the Cartesian coordinates of the simplex vertices

        Returns:
            The volume of the simplex.
        """
        T = np.zeros(self.N_dim)
        T = (vertex_coords[:, 0:self.N_dim]
             - vertex_coords[:, self.N_dim].reshape(-1, 1))

        # https://en.wikipedia.org/wiki/Simplex#Volume
        jac_det = np.absolute(np.linalg.det(T))
        return jac_det / factorial(self.N_dim)

    def rand_stratum(self, N) -> np.ndarray:
        """Generate a uniformly distributed random vector in stratum.

        Notice that we do NOT update the stratum inside this function.
        That is, adding the samples to the stratum needs to be done outside.
        This is done so that one could still do accept/reject type of methods.

        Args:
            N: The number of samples

        Returns:
            A dxN Matrix, where each column is a random vector in this stratum.
        """
        X = np.zeros((self.N_dim, N))

        z0 = self.vertices[:, 0].reshape(-1, 1)
        C = self.vertices[:, 1:] - z0
        X = _rand_simplex(N, self.N_dim, C, z0)

        return X
