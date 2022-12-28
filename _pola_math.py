import numpy as np
from astropy.table import Table, join, Column
import decimal
from numpy.random import normal
from numpy import inf
import pandas as pd

def transposeTable(tab_before, id_col_name='ID'):
    '''
    Returns a copy of tab_before (an astropy.Table) with rows and columns interchanged
    id_col_name: name for optional ID column corresponding to
    the column names of tab_before
    
    Source:
    https://gist.github.com/PBarmby - github, tab_trans.py 
    
    '''
    # contents of the first column of the old table provide column names for the new table
    # TBD: check for duplicates in new_colnames & resolve
    new_colnames=tuple(tab_before[tab_before.colnames[0]])
    # remaining columns of old table are row IDs for new table 
    new_rownames=tab_before.colnames[1:]
    # make a new, empty table
    tab_after=Table(names=new_colnames)
    # add the columns of the old table as rows of the new table
    for r in new_rownames:
        tab_after.add_row(tab_before[r])
    if id_col_name != '':
        # add the column headers of the old table as the id column of new table
        tab_after.add_column(Column(new_rownames, name=id_col_name),index=0)
    return(tab_after)


def Truncate(num, decim):
    '''
    Truncates number to a desired decimal
    
    Input:
    -----
    - num : [float]
    - decimal : [int]
    
    Output:
    --------
    - trunc_num : [float] truncated number 
    
    '''
    d = 10**(-decim)
    trunc_num = float(decimal.Decimal(num).quantize(decimal.Decimal('{}'.format(d)), rounding=decimal.ROUND_DOWN))
    return trunc_num


def Excess_variance(mag, magerr):
    """
    Calculates excess variance as defined by Sanchez et al. 2017

    Input
    -----
    mag: [float]
    magerr: [float]

    Output
    ----
    sigma_rms_sq - sigma_rms_sq_err

    """
    mean_mag = np.mean(mag)
    nobs = len(mag)
    sigma_rms_sq = 1/(nobs * mean_mag**2) * np.sum((mag - mean_mag)**2 - magerr**2) 
    sd = 1/nobs * np.sum((((mag - mean_mag)**2 - magerr**2) - sigma_rms_sq * mean_mag**2 )**2)
    sigma_rms_sq_err = sd / (mean_mag**2 * nobs**1/2)

    return sigma_rms_sq - sigma_rms_sq_err

def FalseSpectrum(df, specname, n):
    '''
    Generates False Spectrum using a simple Monte Carlo sampling where every flux at fixed wavelength
    follows a gaussian distribution with standard deviation equal to the error
    ======
    Input:
    ======
    df [DataFrame] : DataFrame of the spectrum 
    n [int] : Number of false spectra
    ------------------------------------
    ======
    Output:
    ======
    output [dict] : Dictionary with the false spectra
    '''
    output = {}
    while(n>0):
        flux = df.flux_f_syn
        error = df.error
        error[np.isnan(error)] = 0
        error[error == inf] = np.max(flux)*1000
        new_flux = [normal(mu, sigma) for mu, sigma in zip(flux, error)]
        df_aux = df
        df_aux = df_aux.drop(columns= 'flux_f_syn')
        print(type(df_aux))
        df_aux['flux_f_syn'] = new_flux
        output[specname + '_v{}'.format(n)] = df_aux
        n = n-1
    return output 
