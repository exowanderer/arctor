# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).
# Source: https://github.com/pcubillos/mc3
from . import _dwt as dwt


def dwt_chisq(model, data, params, priors=None, priorlow=None, priorup=None):
    """
    Calculate -2*ln(likelihood) in a wavelet-base (a pseudo chi-squared)
    based on Carter & Winn (2009), ApJ 704, 51.

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    params: 1D float ndarray
        Model parameters (including the tree noise parameters: gamma,
        sigma_r, sigma_w; which must be the last three elements in params).
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Left-sided prior standard deviation (param < prior).
        A priorlow value of zero denotes a uniform prior.
    priorup: 1D ndarray
        Right-sided prior standard deviation (prior < param).
        A priorup value of zero denotes a uniform prior.

    Returns
    -------
    chisq: Float
        Wavelet-based (pseudo) chi-squared.

    Notes
    -----
    - If the residuals array size is not of the form 2**N, the routine
    zero-padds the array until this condition is satisfied.
    - The current code only supports gamma=1.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> import numpy as np
    >>> # Compute chi-squared for a given model fitting a data set:
    >>> data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0, 0.0])
    >>> model = np.ones(8)
    >>> params = np.array([1.0, 0.1, 0.1])
    >>> chisq = ms.dwt_chisq(model, data, params)
    >>> print(chisq)
    1693.22308882
    >>> # Now, say this is a three-parameter model, with a Gaussian prior
    >>> # on the last parameter:
    >>> priors = np.array([1.0, 0.2, 0.3])
    >>> plow   = np.array([0.0, 0.0, 0.1])
    >>> pup    = np.array([0.0, 0.0, 0.1])
    >>> chisq = ms.dwt_chisq(model, data, params, priors, plow, pup)
    >>> print(chisq)
    1697.2230888243134
    """
    if len(params) < 3:
        with mu.Log() as log:
            log.error('Wavelet chisq should have at least three parameters.')

    if priors is None or priorlow is None or priorup is None:
        return dwt.chisq(params, model, data)

    iprior = (priorlow > 0) & (priorup > 0)
    dprior = (params - priors)[iprior]
    return dwt.chisq(params, model, data, dprior,
                     priorlow[iprior], priorup[iprior])


def dwt_daub4(array, inverse=False):
    """
    1D discrete wavelet transform using the Daubechies 4-parameter wavelet

    Parameters
    ----------
    array: 1D ndarray
        Data array to which to apply the DWT.
    inverse: bool
        If False, calculate the DWT,
        If True, calculate the inverse DWT.

    Notes
    -----
    The input vector must have length 2**M with M an integer, otherwise
    the output will zero-padded to the next size of the form 2**M.

    Examples
    --------
    >>> import numpy as np
    >>> improt matplotlib.pyplot as plt
    >>> import mc3.stats as ms

    >>> # Calculate the inverse DWT for a unit vector:
    >>> nx = 1024
    >>> e4 = np.zeros(nx)
    >>> e4[4] = 1.0
    >>> ie4 = ms.dwt_daub4(e4, True)
    >>> # Plot the inverse DWT:
    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(np.arange(nx), ie4)
    """
    isign = -1 if inverse else 1
    return dwt.daub4(np.array(array), isign)
