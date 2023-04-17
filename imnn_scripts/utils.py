from functools import partial
from jax import lax
import jax.numpy as jnp
import jax
import jax_cosmo as jc
import numpy as np


@jax.jit
def rotate_sim(k, sim):
    k = k % 4

    condition1 = (k > 0)
    condition2 = (k > 1)
    condition3 = (k > 2)
    condition4 = (k == 3)

    # if k == 0:
    def kzero(k):
        return sim
    # if k == 1:
    def kone(k):
        return jnp.rot90(sim, k=1, axes=(1,2))
    # if k == 2:
    def ktwo(k):
        return jnp.rot90(sim, k=2, axes=(1,2))
    def kthree(k):
        return jnp.rot90(sim, k=3, axes=(1,2))

    # if >2, use kthree, else use ktwo
    def biggerthantwo(k):
        return lax.cond(condition3, true_fun=kthree, false_fun=ktwo, operand=k)

    # if > 1, return biggerthan2, else use kone
    def biggerthanone(k):
        return lax.cond(condition2, true_fun=biggerthantwo, false_fun=kone, operand=k)

    # if >0 , return biggerthan1, else use kzero
    sim = lax.cond(condition1, true_fun=biggerthanone, false_fun=kzero, operand=k)

    return sim



@jax.jit
def compute_variance_catalog(omegaM=0.3175):

    N0 = 64
    N1 = 64
    N2 = 128
    L0 = 1000.
    L1 = 1000.
    L2 = 5500.
    zmean = jnp.array([0.5, 1.0, 1.5, 2.0])
    Ncat = 4

    cosmo = jc.Planck15(Omega_c=omegaM, sigma8=0.8) # no sigma8-dependence 
    rms = 0.3 / 2. # from review (corrected w Hall comment)
    a = 1. / (1. + zmean)
    dist = jc.background.radial_comoving_distance(cosmo, a, log10_amin=-3, steps=256)
    angle = 2. * jnp.arctan((L0/N0/2) / dist)
    arcmin_angle = angle * 180. / np.pi * 60.
    arcmin2_pix = arcmin_angle**2
    sources = 30. / Ncat * arcmin2_pix # change to 30 galaxy/arcmin^2 
    return rms**2 / sources


# make jax version of weiner filter

#@partial(jax.jit, static_argnums=(1,2))
def wiener_jax(im, kernel, noise=None):
    """
    Perform a Wiener filter on an N-dimensional array.
    Apply a Wiener filter to the N-dimensional array `im`.
    Parameters
    ----------
    im : ndarray
        An N-dimensional array.
    mysize : int or array_like, optional
        A scalar or an N-length list giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
        If mysize is a scalar, then this scalar is used as the size
        in each dimension.
    noise : float, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.
    Returns
    -------
    out : ndarray
        Wiener filtered result with the same shape as `im`.
    Notes
    -----
    This is a line-for-line implementation of the scipy.signal.wiener function in Jax.
    """
    #im = jnp.array(im)
    #if mysize is None:
        #mysize = [3] * im.ndim
       
    mysize = jnp.array(kernel.shape)

    if mysize.shape == ():
        mysize = jnp.repeat(mysize.item(), im.ndim)

    # Estimate the local mean
    lMean = jax.scipy.signal.correlate(im, kernel, 'same') / jnp.prod(mysize, axis=0)

    # Estimate the local variance
    lVar = (jax.scipy.signal.correlate(im ** 2, kernel, 'same') /
            jnp.prod(mysize, axis=0) - lMean ** 2)

    # Estimate the noise power if needed.
    if noise is None:
        noise = jnp.mean(jnp.ravel(lVar), axis=0)

    res = (im - lMean)
    res *= (1 - noise / lVar)
    res += lMean
    out = jnp.where(lVar < noise, lMean, res)

    return out


def indices_vector(num_tomo=4):
   """function for getting indices of auto- and cross-spectra
   for num_tomo tomographic bins"""
   indices = []
   cc = 0
   for catA in range(0,num_tomo,1):
      for catB in range(catA,num_tomo,1):
        indices.append([catA, catB])
        cc += 1
   return indices

indices = jnp.array(indices_vector())


@partial(jax.jit, static_argnums=(2,3,4,5,6,7,8))
def compute_pk_2d_jax(field, field2=None, 
                      Nbins=82, 
                      endidx=24,
                      L0=1000.,
                      L1=1000.,
                      L2=5500.,
                      N0=64,
                      N1=64):
    L0 = L0
    L1 = L1
    L2 = L2

    N0 = N0
    N1 = N1

    # N0 = jnp.shape(field)[0]
    # N1 = jnp.shape(field)[1]
    
    dV = L0*L1/(N0*N1)
    shat = jnp.fft.fftn(field)*dV

    if field2 is not None:
        shat2 = jnp.fft.fftn(field2)*dV
    else:
        shat2 = shat

    #P = jnp.real(shat)**2 + jnp.imag(shat2)**2
    #P = jnp.real(P)
    P = jnp.real(shat * jnp.conj(shat2))
    
    ik0 = jnp.fft.fftfreq(N0, d=L0/N0)*2*np.pi
    ik1 = jnp.fft.fftfreq(N1, d=L1/N1)*2*np.pi
    
    #k = jnp.sqrt(ik0[:,None,]**2 + ik1[None,:(N1//2+1)]**2)
    k = jnp.sqrt(ik0[:,None,]**2 + ik1[None,:]**2)

    Pw, _ = jnp.histogram(k, bins=Nbins, range=(0,1))
    
    P, b = jnp.histogram(k, weights=P, bins=Nbins, range=(0,1))
    P /= L0*L1

    P = jnp.where((Pw > 0), P / Pw, P)

    mask = P > 0

    return b[1:][:endidx],P[:endidx]


@jax.jit
def get_auto_and_cross_spec(single_sim, Nbins=82, endidx=24, 
                            indices=indices):
    """get all auto- and cross- power spectra of a complex shear field in Jax"""
    
    _compute_pk = lambda d1,d2: compute_pk_2d_jax(d1, field2=d2, Nbins=Nbins, endidx=endidx)

    # outs will be of shape (num_spec, num_bins_per_spec,)
    pk_outs = jnp.ones((indices.shape[0], endidx))
    k = jnp.ones((endidx,))

    def body_fun(i, inputs):
        idx,data,pk_outs,k = inputs
        k,pk = _compute_pk(data[idx[i, 0], ...], data[idx[i, 1], ...])
        pk_outs = pk_outs.at[i, ...].set(pk)
        k = k.at[...].set(k)
        return idx,data,pk_outs,k
    
    init_val = (indices,single_sim,pk_outs,k)

    idx,single_sim,pk_outs,k = jax.lax.fori_loop(lower=0, upper=indices.shape[0], body_fun=body_fun, init_val=init_val)

    return (pk_outs) 