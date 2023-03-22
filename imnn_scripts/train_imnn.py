import cloudpickle as pickle
import h5py as h5
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import matplotlib.pyplot as plt

from utils import rotate_sim
from nets import *
from imnn_mod import *

import json
import sys,os


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def reshape_data(dat):
    realpart = dat[:, ::2, :, :]
    impart = dat[:, 1::2, :, :]

    return jnp.stack([realpart, impart], axis=-1)

### ------------- READ IN CONFIGS -------------

config_path = sys.argv[1]
with open(config_path) as f:
        configs = json.load(f)

LOAD_MODEL = bool(int(sys.argv[2]))    # should we load a model ?

savedir    = configs["savedir"]     # where to save shit
datadir    = configs["datadir"]     # where shit is saved
modeldir   = configs["modeldir"]
priordir   = configs["priordir"]
do_noise   = bool(configs["do_noise"])

if LOAD_MODEL:
    print("training from loaded model")
    model_name = sys.argv[3]        # 
else:
    model_name = None

print("running imnn training, saving to: ", savedir)

# save the configs file to the savedir
with open(savedir + "configs.json", "w") as outfile:
    json.dump(configs, outfile)

### ------------- LOAD ALL DATA -------------
print("loading data, do noise: ", do_noise)
key = jax.random.PRNGKey(33)
key,rng = jax.random.split(key)

fid = jnp.load(datadir + 'noisefree_fid.npy')
val_fid = jnp.load(datadir + 'noisefree_val_fid.npy')
dervs = jnp.load(datadir + 'noisefree_derv.npy')
val_dervs = jnp.load(datadir + 'noisefree_val_derv.npy')

# swap some sims around from train and validation

fid1 = jnp.concatenate([val_fid, fid], axis=0)
idx = jax.random.shuffle(key, jnp.arange(2000), axis=0)

val_fid = val_fid.at[:].set(fid1[idx[1000:]])
fid = fid.at[:].set(fid1[idx[:1000]]); del fid1

dervs1 = jnp.concatenate([dervs, val_dervs])
# chunk into groups of 4: (om-,s8-,om+,s8+)
dervs1 = jnp.array(jnp.split(dervs1, 500))
print('split dervs shape', dervs1.shape)
idx = jax.random.shuffle(rng, jnp.arange(500), axis=0)

val_dervs = val_dervs.at[:].set(jnp.concatenate(dervs1[idx[:250]]))
dervs = dervs.at[:].set(jnp.concatenate(dervs1[idx[250:]])); del dervs1

fid = reshape_data(fid) 
val_fid = reshape_data(val_fid)
dervs = reshape_data(dervs)
val_dervs = reshape_data(val_dervs)


### ------------- IMNN PARAMETERS -------------
θ_fid = jnp.array([0.3175, 0.800]) # CHANGE TO OmegaM=0.6
δθ = 2*jnp.array([0.05, 0.015])

θ_der = (θ_fid + jnp.einsum("i,jk->ijk", jnp.array([-1., 1.]), jnp.diag(δθ) / 2.)).reshape((-1, 2))

n_summaries = 2

n_s = 1000
n_d = 250

n_params = 2
n_summaries = n_params


### ------------- NEURAL NETWORK MODEL -------------

filters = (int(configs["filters"]),)*4
net_scaling = float(configs["net_scaling"])
patience = configs["patience"]
noise_scale = configs["noise_scale"]
act = configs["act"]

if not do_noise:
    net_scaling /= 10.
else:
    net_scaling *= noise_scale



model = CNNRes3D(filters=filters, div_factor=net_scaling, out_shape=2, act=act)
key = jax.random.PRNGKey(42)

input_shape = (4, 64, 64, 2)

### ------------- DEFINE DATA AUGMENTATION SCHEME -------------
### ADD IN NOISE ON TOP OF FIELD SIMS
### NOISE VARIANCES == SIGMA^2

#if do_noise:
noise_variances = jnp.array([1.79560224e-06, 5.44858988e-06, 9.45448781e-06, 1.32736252e-05])
    
#else:
#    noise_variances = jnp.array([0., 0., 0., 0.])

@jax.jit
def noise_simulator(key, sim):
    key1,key2 = jax.random.split(key)
    # do rotations of simulations
    k = jax.random.choice(key1, jnp.array([0,1,2,3]), shape=())
    sim = rotate_sim(k, sim)

    # now add noise
    # this generates white noise across all pixels and then increases the amplitude
    # add zero noise for no-noise case
    sim += (jax.random.normal(key2, shape=(4,64,64,2)) * noise_scale * jnp.sqrt(noise_variances).reshape(4,1,1,1))
    return sim


#### ------------- SET UP IMNN -------------

optimiser = optax.adam(learning_rate=1e-4)

model_key = jax.random.PRNGKey(42)
rng, key = jax.random.split(key)


IMNN = NoiseNumericalGradientIMNN(
    n_s=n_s, n_d=n_d, n_params=n_params, n_summaries=n_summaries,
    input_shape=(4, 64, 64, 2), θ_fid=θ_fid, δθ=δθ, model=model,
    optimiser=optimiser, key_or_state=jnp.array(model_key),
    noise_simulator=jax.jit(lambda rng, d: noise_simulator(
            rng, d)),
    fiducial=fid, 
    derivative=dervs.reshape(n_d, 2, 2, 4, 64, 64, 2),
    validation_fiducial=val_fid,
    validation_derivative=val_dervs.reshape(n_d, 2, 2, 4, 64, 64, 2),
    dummy_graph_input=None,  # dummy graph input
    no_invC=False,
    do_reg=True,
    evidence=False)

if LOAD_MODEL:
    print("LOADING MODEL FROM ", modeldir + model_name)
    loaded_w = load_obj(modeldir + model_name)
    IMNN.w = loaded_w
    IMNN.set_F_statistics(loaded_w, key=model_key)

else:
    pass

## ------------- TRAIN THE IMNN -------------
np=jnp

print("commencing IMNN training")

key,rng = jax.random.split(key)
IMNN.fit(10.0, 0.1, γ=1.0, rng=jnp.array(rng), patience=patience, min_iterations=2000)

print('final IMNN F: ', IMNN.F)
print('final IMNN det F: ', jnp.linalg.det(IMNN.F))

save_obj(IMNN.w, savedir + 'IMNN_w')
jnp.save(savedir + "IMNN_F", IMNN.F)
save_obj(IMNN.history, savedir + 'IMNN_history')

### ------------- PASS PRIOR DATA THROUGH IMNN FUNNEL -------------
np = jnp
#dat = jnp.load("/data80/makinen/borg_sims_fixed/uniform_prior_sims/noisefree_prior_sims.npy")
#params = jnp.load("/data80/makinen/borg_sims_fixed/uniform_prior_sims/noisefree_prior_params.npy")
dat = jnp.load(priordir + "noisefree_prior_sims.npy")
params = jnp.load(priordir + "noisefree_prior_params.npy")


dat = reshape_data(dat)[:, :, :, :, :]

# ADD NOISE TO DATA
noisekey = jax.random.PRNGKey(11)
noisekeys = jax.random.split(noisekey, num=dat.shape[0])
dat = jax.vmap(noise_simulator)(noisekeys, dat)

x1 = IMNN.get_estimate(dat[:2500])
x2 = IMNN.get_estimate(dat[2500:5000])
x3 = IMNN.get_estimate(dat[5000:7500])
x4 = IMNN.get_estimate(dat[7500:])
x = jnp.concatenate([x1,x2,x3,x4])


# save the prior sims' mapping
jnp.save(savedir + "x_imnn.npy", x)
jnp.save(savedir + "theta.npy", params)


#### ------------- get Natalia's target data WITH NOISE -------------
print("loading mock data for obtaining estimates")

import numpy as onp
np = onp
import h5py as h5

def get_data(Ncat,N0,N1,f):
    dataR = onp.zeros((Ncat,N0,N1))
    dataI = onp.zeros((Ncat,N0,N1))
    
    for cat in range(0,Ncat,1):
        survey = f['lensing_catalog_'+str(cat)]['lensing_data']['lensing'][:]
        Nobs = len(survey)
        N0 = f['scalars/N0'][0]
        N1 = f['scalars/N1'][0]
        
        count = np.zeros((N0,N1))
        for nobs in range(0,Nobs,1):
            lens = survey[nobs]
            n0 = int(lens['phi'])
            n1 = int(lens['theta'])
            dataR[cat,n0,n1] = lens['shearR']
            dataI[cat,n0,n1] = lens['shearI']
            count[n0,n1] += 1
    return dataR, dataI


#path = '/data80/nporqueres/borg_sims_fixed/'

path = configs["target_path"]
f = h5.File(path + 'mock_data.h5', 'r')

Ncat = f['scalars/NCAT'][0]

N0 = f['scalars/N0'][0]
N1 = f['scalars/N1'][0]
N2 = f['scalars/N2'][0]

L0 = f['scalars/L0'][0]
L1 = f['scalars/L1'][0]
L2 = f['scalars/L2'][0]

targetR, targetI = get_data(Ncat, N0, N1, f)

_dat = onp.ones((8, 64, 64))
_dat[::2, :, :] = targetR
_dat[1::2, :, :] = targetI

np=jnp
_dat = jnp.array(_dat)
target_data = jnp.squeeze(reshape_data(_dat[jnp.newaxis, :, :, :]))

estimates = IMNN.get_estimate(jnp.expand_dims(target_data, 0))

print('IMNN estimates for target sim', estimates)

jnp.save(savedir + 'estimates', estimates)
