#!/usr/bin/env python

'''


From 

https://docs.astropy.org/en/stable/modeling/example-fitting-line.html
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from scipy.optimize import curve_fit


def fit_line_sigma_clip(x,y,yunc=None,slope=1,intercept=0,niter=5,sigma=3):
    '''
    INPUT
    * x - array of x values
    * y - array of y values

    OPTIONAL INPUT
    * yunc - uncertainty in y values
    * slope - intial guess for slope
    * intercept - intial guess for intercept
    * niter - number of sigma clipping iterations; default is 3
    * sigma - sigma to use in clipping; default is 3

    RETURNS
    * Linear1D model, with slope and intercept
      * you can get fitted y values with fitted_line(x)
    * mask - array indicating the points that were cut in sigma clipping process

    REFERENCES
    https://docs.astropy.org/en/stable/modeling/example-fitting-line.html
    '''
    if yunc is None:
        yunc = np.ones(len(y))
    # initialize a linear fitter
    fit = fitting.LinearLSQFitter()

    # initialize the outlier removal fitter
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=niter, sigma=sigma)

    # initialize a linear model
    line_init = models.Linear1D(slope=slope,intercept=intercept)

    # fit the data with the fitter
    fitted_line, mask = or_fit(line_init, x, y, weights=1.0/yunc)

    return fitted_line, mask
