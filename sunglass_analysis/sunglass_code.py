from scipy import ndimage
import numpy as np
import h5py as h5
import pylab as plt
import healpy as hp
import math
import scipy.stats as stats
from scipy import ndimage

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.cosmology import wCDM, z_at_value
import multiprocessing as mp
import pylab as plt 

import os,sys
import time




# Analysis parameters:

N        = 128
Npix_los = 256

process_particles = True  # Assigns particles to pixels if True (slow); reads file if False.

#---------------------------------------------------
L        = 4000
halfN    = int(N/2+0.5)
res      = L / N
origin   = np.array([-L/2., -L/2., 0.])

print(L, N)

# Cosmology: extract from file

# CONSTANTS
h        = 0.677
c        = 3e5


# run these two things outside and load in to speed things up

comoving_grid = np.arange(1e-3, 4500, 0.1)

# load z table so we don't have to recompute it every time
# try:
#     z_table = np.load("z_table.npy")
#     print("loaded z_table file")
    
# except:
#     print("z_table file not found... recomputing (this may take a while)")
#     z_table = z_at_value(astropy_cosmo.comoving_distance, comoving_grid * u.Mpc / 0.677, method='bounded').value


#---------------------------------------------------


def redshift_from_comoving_distance(comoving_dist, z_table, comoving_grid):
    redshift = np.interp(comoving_dist, comoving_grid, z_table)
    return redshift

def Jacobian(redshift, omega_m):
  E = np.sqrt(omega_m * (1+redshift)**3 + (1-omega_m))
  return 100 * E / c

def weights(z, mean=1.0, std=0.14):
  exponent = - 0.5 * ((z - mean) / std)**2
  norm = std * (2. * np.pi)**0.5
  return np.exp(exponent) / norm

def vox2mph(vox_idx, res, origin):
  return vox_idx * res + origin   
  
def Mpc2spherical(x,y,z):
  r = (x**2 + y**2 + z**2)**0.5
  x_angle = np.arcsin(x / r)
  y_angle = np.arcsin(y / r)  
  return r, x_angle, y_angle

def angular_coord_healpy(index, nside):
  theta, phi = hp.pix2ang(nside=nside, ipix=index)
  return theta, phi

def angular_coord_particles(part, origin):
  x = part[:,0] + origin[0]
  y = part[:,1] + origin[1]
  z = part[:,2] + origin[2]

  r, x_angle, y_angle = Mpc2spherical(x,y,z)
    
  print('Min and max values of x:',np.min(x),np.max(x))
  return r, x_angle, y_angle


def kernel(r, rs, fact, scale_factor):
  K = fact * (rs - r) * r / rs / scale_factor
  return K

def get_kappa_cls_sunglass(fname, outname, infolder, outfolder):
    
    kappa_outname = outfolder + 'kappa/' + outname
    Cl_outname = outfolder + 'Cls/' + outname
    
    simulation_filepath = infolder + fname
    
    print("loading particle simulation from", simulation_filepath)
    print("saving kappa maps to ", kappa_outname)
    print("saving kappa Cls to ", Cl_outname)

    # Load the particle data:

    f = h5.File(simulation_filepath, 'r')
    
    # EXTRACT COSMOLOGY
    print("extracting cosmology from file")
    om_idx = 2
    s8_idx = 9
    
    omega_m = f['scalars/cosmo'][:][0][om_idx]
    sigma_8 = f['scalars/cosmo'][:][0][s8_idx]
    
    print("----\nthis simulation is at \nomega_m = %.3f \nsigma_8 = %.3f \n----"%(omega_m, sigma_8))
    
    print("calculating z table at current cosmology")
    fact = 1.5 * 100**2 * omega_m / c**2
    astropy_cosmo = FlatLambdaCDM(H0=h * 100 * u.km / u.s / u.Mpc, Om0=omega_m, Ob0=0.049)
    
    comoving_grid = np.arange(1e-3, 4500, 0.1)
    z_table = z_at_value(astropy_cosmo.comoving_distance, comoving_grid * u.Mpc / 0.677, method='bounded').value
    

    print('Particles loaded')

    # Explore the file:
    
    part = f['u_pos'][:]

    r, x_angle, y_angle = angular_coord_particles(part, origin)

    Npart = len(part)
    print('Number of particles:',Npart)
    n_average = Npart / L**3  

    x = part[:,0] + origin[0]
    y = part[:,1] + origin[1]
    z = part[:,2] + origin[2]

    # Plot the nearby particles:

    rmax = 120
    

    # Assign particles to voxels in 3D (anglex,angley,radius):

    time0 = time.time()

    anglemax  = np.arctan(0.5)
    dangle    = anglemax/halfN

    # coord contains various particle coordinates:
    coord      = np.empty((Npart,6))

    coord[:,0] = r
    coord[:,1] = x_angle
    coord[:,2] = y_angle
    coord[:,3] = redshift_from_comoving_distance(r, z_table, comoving_grid)
    coord[:,4] = x_angle/dangle + halfN  # x position (index)
    coord[:,5] = y_angle/dangle + halfN  # y position (index)

    print('Coordinates assigned')

    print('Time:',time.time()-time0)

    new_pixel_count = np.zeros((N+1,N+1,Npix_los*3),dtype=int)
    xpos            = np.empty(Npart,dtype=int)
    ypos            = np.empty(Npart,dtype=int) 
    kappa           = np.zeros((N,N,Npix_los))
    radial_bin      = np.zeros(Npart,dtype=int)
    
    print("KAPPA SHAPE: ", kappa.shape)


    # Radii of thin radial shells (on which to compute kappa, which is then [weighted] summed to do tomography):
    r_pix_los = np.arange(0, Npix_los, 1) * L / Npix_los
    dr        = L/Npix_los
    redshift  = redshift_from_comoving_distance(r_pix_los, z_table, comoving_grid)

    time2 = time.time()

    xpos[:]       = np.round(coord[:,4]).astype(int)
    ypos[:]       = np.round(coord[:,5]).astype(int)

    # Put particles that are out of the field (either on the sky, or too far away) on to the edges/furthest slice 
    # (i.e. elements labelled by any of [N,N,Npix_los] are excluded particles, ignored later):

    radial_bin[:] = np.where(coord[:,0]<L,np.round(coord[:,0]/dr).astype(int),Npix_los)

    xpos[:] = np.where(xpos[:] > -1, xpos[:], N)
    xpos[:] = np.where(xpos[:] <  N, xpos[:], N)

    ypos[:] = np.where(ypos[:] > -1, ypos[:], N)
    ypos[:] = np.where(ypos[:] <  N, ypos[:], N)

    # Compute the convergence on spherical shells (they *are* spherical - the last index is a radius)

    # Improve this later: the volume of each narrow pyramid changes slightly across the box

    approx_solid_angle = (np.arctan(0.5)/halfN)**2

    # Note the weight include dr and the Jacobian from z to r:

    w = weights(redshift) * Jacobian(redshift, omega_m) * dr

    # Declare arrays to do the Sunglass summation:

    dSum1 = np.zeros((N+1,N+1,Npix_los+1))
    dSum2 = np.zeros((N+1,N+1,Npix_los+1))
    Sum1  = np.zeros((N+1,N+1,Npix_los))
    Sum2  = np.zeros((N+1,N+1,Npix_los))

    # Overall amplitude to change sums to kappa:

    factor = fact/n_average/approx_solid_angle
    
    
    # This might not be very efficient because of for loops.  The sum over particles (part) is the slowest step. 

    # dSum1 and dSum2 collect the contributions to the sunglass sums from particles in the narrow r ranges.
    # Sum1 and Sum2 then add them (rather inefficiently, but this a small calculation)

    for part in range(Npart): 

        oneplusz = 1.+coord[part,3]
        ix = xpos[part]
        iy = ypos[part]
        jbin = radial_bin[part]

        dSum1[ix,iy,jbin]  += oneplusz/coord[part,0]
        dSum2[ix,iy,jbin]  += oneplusz

    # This line can be removed for speed but can be a useful diagnostic - the number of particles per pixel shouln't
    # vary systematically with position in the grid.
    #    new_pixel_count[ix,iy,jbin] += 1

    # kappa contains the convergence field on spherical shells, finely-spaced (shell j indexed last)

    for j in range(1, Npix_los):
        Sum1[:,:,j]=np.sum(dSum1[:,:,0:j],axis=2)
        Sum2[:,:,j]=np.sum(dSum2[:,:,0:j],axis=2)

        kappa[:,:,j] =  factor*(Sum1[:N,:N,j]-Sum2[:N,:N,j]/r_pix_los[j])

        # Subtract the mean:
        kappa[:,:,j] += -np.average(kappa[:,:,j])

    time3 = time.time()

    print('Time (new method) =',time3-time2)
    
    # Calculate kappa for a tomographic bin (kappa_tomo) by integrating over the radial window.
    # w includes the n(z) weight, the Jacobian, and the element dr.
    
    print("w", w.shape)
    print("kappa", kappa.shape)
    
    kappa_tomo = np.zeros((N,N))
    
    for i in range(0,N):
      for j in range(0,N):
        kappa_tomo[i,j] = np.sum(kappa[i,j,:] * w)

    # Compute the average empirically, and subtract

    kappa_tomo=kappa_tomo - np.average(kappa_tomo)
    

    # Next output the Cls

    # Calculate power spectrum of the tomographic field.
    # This is inherited from a separate code.  

    # Size of field and various derived quantities:

    field_size_rad    = 2.0 * np.arctan(0.5)

    l_fundamental     = 2.* np.pi / field_size_rad
    density_of_states = 1.0/l_fundamental**2

    print('Fundamental       =', l_fundamental)
    print('Density of states =', density_of_states)

    # Fourier Transform

    # Round up the size along this axis to an even number
    n = int( math.ceil(kappa_tomo.shape[0] / 2.) * 2 )
    npix = n*n

    print('Number of pixels on a side:',kappa_tomo.shape[0],n)

    ft      = np.fft.fftn(kappa_tomo)
    power2d = np.abs(ft)**2 / npix**2   

    image_variance = ndimage.variance(kappa_tomo)

    print('Variance of image    =', image_variance)

    # Find frequencies

    kvec = np.fft.fftfreq(n,d=1.0) * n

    kvec2D = np.meshgrid(kvec, kvec)

    ell = np.sqrt(kvec2D[0]**2 + kvec2D[1]**2) * l_fundamental
    ell = ell.flatten()

    power=power2d.flatten()

    print('Parseval variance    =',np.sum(power))

    # Construct bins in ell space (separated by the fundamental wavenumber):

    ell_bins = np.arange(0.5, n//2+1, 1.) * l_fundamental
    ell_vals = 0.5 * (ell_bins[1:] + ell_bins[:-1])

    binned_power, bin_edges, bin_number = stats.binned_statistic(ell, power, statistic = "mean", bins = ell_bins)

    D_ell = binned_power * (2.* ell_vals+1) * field_size_rad**2 /(4.*np.pi)

    bin_width = bin_edges[1]-bin_edges[0]

    D_ell_variance = np.sum(D_ell*bin_width)

    print('D_ell variance       =',D_ell_variance)

    # Plot power spectrum:

    X_ell = D_ell * ell_vals *(ell_vals + 1) / (2.*ell_vals) * 2.
    C_ell = binned_power * field_size_rad**2

    Cl_output = np.stack([ell_vals, C_ell], -1)
    np.save(Cl_outname, Cl_output)