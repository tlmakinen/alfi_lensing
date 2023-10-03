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

    # Assign particles to pixels, or read in from file.

    if(process_particles):

        anglemax  = np.arctan(0.5)
        dangle    = anglemax/halfN

        coord = np.empty((Npart,7))
        pixel_particles = np.empty((N,N,6,int(1.2*Npart/(N*N))))
        pixel_count = np.zeros((N,N),dtype=int)

        coord[:,0] = r
        coord[:,1] = x_angle
        coord[:,2] = y_angle
        coord[:,3] = redshift_from_comoving_distance(r, z_table, comoving_grid)
    #    coord[:,4] = np.tan(x_angle)/dtan + halfN
    #    coord[:,5] = np.tan(y_angle)/dtan + halfN 
        coord[:,4] = x_angle/dangle + halfN
        coord[:,5] = y_angle/dangle + halfN 

    # Find a faster way than this!
    # Also, it leaves an empty row and column.

        for p in range(Npart):
            if(np.mod(p,10000000)==0):
                print(p)
            ix = int(coord[p,4])
            iy = int(coord[p,5])

    # Count if it the pixel index is actually in the box, and throw out particles with r>L:
    # I think ix=0 and iy=0 remain empty (fix this later)

            if(np.abs(ix-halfN)<halfN):
                if(np.abs(iy-halfN)<halfN):
                    if(coord[p,0]<L):
                        pixel_particles[ix,iy,0:4,pixel_count[ix,iy]]=coord[p,0:4]
                        pixel_count[ix,iy]+=1

        print('Max number of particles in a pixel:',np.max(pixel_count))

    # The sorting doesn't actually help yet, but could streamline the sums later:

        for ix in range(N):
            for iy in range(N):

                np.ndarray.sort(pixel_particles[ix,iy,:,0:pixel_count[ix,iy]],axis=1)

        #np.save("Sorted_"+str(L)+"_"+str(N)+"_"+str(Npix_los)+"_lightcone_particles", pixel_particles[:,:,:,0:np.max(pixel_count)])
        #np.save("Sorted_"+str(L)+"_"+str(N)+"_"+str(Npix_los)+"_lightcone_count", pixel_count)

        #print("Sorted catalogue saved")
        
    else:   

    # I don't think this is working:

        print('Loading catalogue')

        pixel_particles = np.load("Sorted_"+str(L)+"_"+str(N)+"_"+str(Npix_los)+"_lightcone_particles.npy") 
        pixel_count = np.load("Sorted_"+str(L)+"_"+str(N)+"_"+str(Npix_los)+"_lightcone_count.npy") 

        print('Catalogue loaded')


    # Compute the convergence on spherical shells (they are spherical - the last index is a radius)

    # Improve this later: the volume of each narrow pyramid changes slightly across the box

    approx_solid_angle = (np.arctan(0.5)/halfN)**2

    kappa = np.zeros((N,N,Npix_los))

    r_pix_los = np.arange(0, Npix_los, 1) * L / Npix_los
    dr        = L/Npix_los
    redshift  = redshift_from_comoving_distance(r_pix_los, z_table, comoving_grid)

    print("distance", np.min(r_pix_los), np.max(r_pix_los))
    print("redshift", np.min(redshift), np.max(redshift))

    # Note change in w to include dr

    w = weights(redshift) * Jacobian(redshift, omega_m) * dr

    dSum1 = np.zeros(Npix_los)
    dSum2 = np.zeros(Npix_los)
    Sum1  = np.zeros(Npix_los)
    Sum2  = np.zeros(Npix_los)

    # This might not be very efficient because of for loops. The particles are ordered in increasing r, 
    # which might also be useful.

    # dSum1 and dSum2 collect the contributions to the sunglass sums from particles in the narrow r ranges.
    # Sum1 and Sum2 then add them (rather inefficiently, but it's a small calculation)

    for ix in range(N):
        for iy in range(N):

            dSum1[:] = 0.0
            dSum2[:] = 0.0

            rpart = pixel_particles[ix,iy,0,:]
            zpart = pixel_particles[ix,iy,3,:]
            apart = 1./(1.+zpart)

            for p in range(pixel_count[ix,iy]):
                jbin = int(rpart[p]/dr)
                dSum1[jbin] += 1./(rpart[p]*apart[p])
                dSum2[jbin] += 1./apart[p]

            for j in range(1, Npix_los):
                Sum1[j]=np.sum(dSum1[0:j])
                Sum2[j]=np.sum(dSum2[0:j])

                rj = r_pix_los[j]
                kappa[ix,iy,j] =  fact*(Sum1[j]-Sum2[j]/rj ) /n_average / approx_solid_angle


    # Calculate kappa for a tomographic bin by integrating over the radial window.
    # w includes the weight, the Jacobian, and the element dr.

    kappa_tomo = np.zeros((N,N))
    for i in range(0,N):
      for j in range(0,N):
        kappa_tomo[i,j] = np.sum(kappa[i,j,:] * w)

    # Compute the average empirically, and subtract (note that there is some large-scale
    # feature that is probably related to sinc dependence of pyramid area.  Fix later):

    kappa_tomo=kappa_tomo - np.average(kappa_tomo)


    np.savez(kappa_outname, kappa_tomo=kappa_tomo, kappa=kappa)


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