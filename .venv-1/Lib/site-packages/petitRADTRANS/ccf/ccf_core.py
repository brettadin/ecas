"""Core cross-correlation-related functions.
"""

import numpy as np
from scipy.interpolate import interp1d


def co_add_cross_correlation(cross_correlation, velocities_ccf, co_added_velocities):
    n_co_additions = np.shape(cross_correlation)[0]

    if np.ndim(co_added_velocities) != 3:
        raise ValueError(f"co-added velocities must have 3 dimensions, but has {np.ndim(co_added_velocities)}")
    elif np.shape(co_added_velocities)[1] != n_co_additions:
        raise ValueError(f"size of co-added velocities dimension 1 must be the size of cross correlation dimension 0 "
                         f"({n_co_additions}), but is of shape {np.shape(co_added_velocities)}")

    co_added_ccf = np.zeros((np.shape(co_added_velocities)[0], np.shape(co_added_velocities)[2]))  # (Kp, Vr)

    # Interpolate the CCF at a given integration over the new (Kp, Vr) space
    ccf_interpolation_fs = [interp1d(velocities_ccf, ccf) for ccf in cross_correlation]

    for i, v_rest in enumerate(co_added_velocities):
        for j in range(n_co_additions):
            co_added_ccf[i, :] += ccf_interpolation_fs[j](v_rest[j, :])  # sum every integration at a given Kp

    return co_added_ccf


def cross_correlate(array_1, array_2):
    """Cross-correlate two 1-D arrays.

    Args:
        array_1: first array
        array_2: second array

    Returns:
        The cross-correlation values.
    """
    array_1_2 = np.dot(array_1, array_1)
    array_2_2 = np.dot(array_2, array_2)
    array_dot = np.dot(array_1, array_2)

    return array_dot / np.sqrt(array_1_2 * array_2_2)


def cross_correlate_matrices(matrix_1, matrix_2):
    """Cross-correlate two N-D arrays.
    The cross-correlation is performed over the last dimension of the matrices. NaNs are ignored and replaced by zeros.
    For matrices of shape e.g. (2, 3, 10, 1000), the result is a matrix of shape (2, 3, 10). This is equivalent, but
    faster, to:
    >>> ccf = np.zeros((2, 3, 10))
    >>> for i in range(2):
    ...     for j in range(3):
    ...         for k in range(10):
    ...             ccf[i, j, k] = cross_correlate(matrix_1[i, j, k, :], matrix_2[i, j, k, :])

    For 1-D arrays, cross_correlate is faster, but NaNs are not treated.
    Arguments matrix_1 and matrix_2 can be 1-D arrays, in that case NaNs are treated then cross_correlate() is used. The
    NaN treatment is rather slow, so cross_correlate is still faster if no NaN are in the arrays.

    Args:
        matrix_1: first matrix
        matrix_2: second matrix

    Returns:
        The cross-correlation values.
    """
    matrix_dot = matrix_1 * matrix_2

    # Ignore indices where both matrices are NaNs: any number times NaN will return NaN, hence matrix_dot can be used
    indices = np.isnan(matrix_dot)

    matrix_1[indices] = 0  # replace NaNs with zeros, zeros won't add to the sum
    matrix_2[indices] = 0

    if np.ndim(matrix_1) <= 1:  # for 1-D arrays, cut-off to the faster cross_correlate, after the NaN treatment
        return cross_correlate(matrix_1, matrix_2)  # slightly inefficient as matrix_dot is not used

    matrix_dot[indices] = 0

    # Intermediate calculations, using einsum is ~2 times faster than np.sum(matrix ** 2, axis=-1)
    matrix_1 = np.einsum(matrix_1, [Ellipsis, 0], matrix_1, [Ellipsis, 0])  # <=> np.sum(matrix_1 ** 2, axis=-1)
    matrix_2 = np.einsum(matrix_2, [Ellipsis, 0], matrix_2, [Ellipsis, 0])  # <=> np.sum(matrix_2 ** 2, axis=-1)

    matrix_dot = np.sum(matrix_dot, axis=-1)

    # Use matrix 1 to build the divisor in order to save RAM
    matrix_1 *= matrix_2
    matrix_1 = np.sqrt(matrix_1)

    # Use matrix 2 to store the result in order to save RAM
    matrix_2.fill(0.0)

    indices = np.nonzero(matrix_1)  # prevent division by 0

    # Cross correlation
    # result = matrix_1 . matrix_2 / sqrt(matrix_1 . matrix_1 * matrix_2 . matrix_2)
    matrix_2[indices] = matrix_dot[indices] / matrix_1[indices]

    return matrix_2
