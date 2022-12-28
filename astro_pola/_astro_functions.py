import numpy as np 
import astropy 






def FluxToMag(flux, fluxerr, magzero=24.3):
    """
    Transform flux and fluxerr to magnitude 
    
    Input
    -----
    flux :
    fluxerr :
    magzero : 

    Output:
    ------
    mag, magerr
    """
    mag = -2.5*np.log10(flux) + magzero
    magerr = np.fabs(-2.5 * fluxerr/(flux * np.log(10)))
    return mag, magerr


def FluxJyToABMag(flux, fluxerr=None):
    """
    Transform flux and fluxerr in Jy to AB magnitude 
    
    Input
    -----
    flux :
    fluxerr :

    Output:
    ------
    magab, magab_err
    """
    magab = -2.5*np.log10(flux) + 8.9
    magab_err = None
    if type(fluxerr)==float:
        magab_err = np.fabs(-2.5 * fluxerr/(flux * np.log(10)))

    return magab, magab_err


def ABMagToFluxJy(mab, mab_err=None):
    """
    Returns AB magnitude to flux in Jy

    Input:
    -----
    mab : [float]

    Output:
    -------
    f : [float]
    """
    f = 10**(-0.4*(mab - 8.90))
    if type(mab_err) != type(None):
        f_err = np.sqrt((f*np.log(10)*0.4*mab_err)**2)
        return f, f_err
    else:
        return f