import numpy as onp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax_cosmo as jc

import cloudpickle as pickle

from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u


### FIXED ANALYSIS VALUES

comoving_grid = np.arange(1e-3, 4500, 0.1)




parameters = ["omegaM", "sigma8"]
value_headings = ["fid", "minus", "plus"]

param_values = {
    "omegaM": [float(θ_fid[0]), float(θ_fid[0] - δθ[0]), float(θ_fid[0] + δθ[0])],
    "sigma8": [float(θ_fid[1]), float(θ_fid[1] - δθ[1]), float(θ_fid[1] + δθ[1])]
}

parameter_table = {}

for i,p in enumerate(parameters):
    for j,v in enumerate(value_headings):
        parameter_table[p] = dict(zip(value_headings, (param_values[p])))



def redshift_from_comoving_distance(comoving_dist, z_table, comoving_grid):
    redshift = np.interp(comoving_dist, comoving_grid, z_table)
    return redshift


def get_redshift_table(omega_m,
                       N=128,
                       Npix_los=256,
                       L=4000
                      ):
    
    """
    Get redshift values that correspond to the given line-of-sight voxel.
    THIS IS A VERY SLOW CODE DUE TO INTERPOLATION TIME IN `astropy_cosmo`.
    """
                    
    halfN    = int(N/2+0.5)
    res      = L / N
    origin   = np.array([-L/2., -L/2., 0.])
    dr        = L/Npix_los

    # CONSTANTS
    h        = 0.677
    c        = 3e5

    astropy_cosmo = FlatLambdaCDM(H0=h * 100 * u.km / u.s / u.Mpc, Om0=omega_m, Ob0=0.049)

    comoving_grid = np.arange(1e-3, 4500, 0.1)
    z_table = z_at_value(astropy_cosmo.comoving_distance, comoving_grid * u.Mpc / 0.677, method='bounded').value

    # Radii of thin radial shells (on which to compute kappa, which is then [weighted] summed to do tomography):
    r_pix_los = np.arange(0, Npix_los, 1) * L / Npix_los
    dr        = L/Npix_los
    return redshift_from_comoving_distance(r_pix_los, z_table, comoving_grid)




def Jacobian(redshift, omega_m):
    """
    Cosmological Jacobian for tomographic bin calculation
    """
    E = np.sqrt(omega_m * (1+redshift)**3 + (1-omega_m))
    return 100 * E / c

def weights(z, mean=1.0, std=0.14):
    """"
    Calculate weighting at given redshift value
    and mean of tomographic bin
    """
    exponent = - 0.5 * ((z - mean) / std)**2
    norm = std * (2. * np.pi)**0.5
    return np.exp(exponent) / norm


def get_tomo_bin(kappa, 
                z_table, 
                omega_m,
                z_mean,
                N=128,
                Npix_los=256,
                L=4000
                ):
    """
    Calculate tomographic bin from existing sunglass kappa field and pre-calculated z_table
    kappa: array_like
        full (N,N,Npix_los) kappa field to be put into tomographic bins
    z_table: array_like
        (Npix_los,) array of redshift values per line-of-sight voxel
    omega_m: float
        value at which to compute the cosmological Jacobian for the tomographic bin
    N: int
        image size (N,N)
    Npix_los: int
        line-of-sight voxel number (in redshift direction)
    L: int, box size
        physical box size in redshift direction (Mpc)
    """
    
    
    dr = L/Npix_los
    # broadcast over z-array in LOS direction
    w = weights(z_table, mean=z_mean) * Jacobian(z_table, omega_m) * dr

    kappa_tomo = np.zeros((N,N))

    for i in range(0,N):
      for j in range(0,N):
        kappa_tomo[i,j] = np.sum(kappa[i,j,:] * w)

    # Compute the average empirically, and subtract

    return kappa_tomo - np.average(kappa_tomo)
