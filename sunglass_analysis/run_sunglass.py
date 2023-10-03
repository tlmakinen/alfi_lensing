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

from sunglass_code import *


import cloudpickle as pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# ------ DEFINE BATCH SIZE

#batch_size = 5

# ------

# STRATEGY
# save filename list for each folder and set up a batch index to do groups of 100 or so in parallel
# which means we need


folder = sys.argv[1]
batch_idx = int(sys.argv[2])
batch_size = int(sys.argv[3])
load_filenames = bool(int(sys.argv[4]))

outfolder = "/data101/makinen/borg_sunglass/" + folder + "/"

sim_path = "/data80/nporqueres/sims_borg_forward/" + folder + "/"


simnames = load_obj(folder + "_fnames.pkl")
sim_outnames = load_obj(folder + "_fnames_out.pkl")


derivatives_batch_idx = np.arange(0, 1000, step=batch_size)
fiducial_batch_idx = np.arange(0, 2000, step=batch_size)
prior_batch_idx = np.arange(0, 10000, step=batch_size)

# WHICH FOLDER ARE WE USING ?
if folder == "fiducial":
    batch_indices = fiducial_batch_idx

elif folder == "prior":
    batch_indices = prior_batch_idx

else:
    batch_indices = derivatives_batch_idx


BATCH_START = batch_indices[batch_idx]
BATCH_END   = batch_indices[batch_idx + 1]

simnames = simnames[BATCH_START:BATCH_END]
sim_outnames = sim_outnames[BATCH_START:BATCH_END]

# ----- START THE STUPID FOR-LOOP ------

print("RUNNING BATCH %d"%(batch_idx), "FOR THE %s"%(folder), "SUITE")

t1 = time.time()
counter = 0



for i,fname in enumerate(simnames):

    print("PROCESSING SIMULATION %d OF %d"%(i+1, len(simnames)))
    print("-----")

    outname = sim_outnames[i]

    get_kappa_cls_sunglass(fname, outname, infolder=sim_path, outfolder=outfolder)

    counter += 1



t2 = time.time()

print("time to process %d sims: "%(counter + 1), t2 - t1, " seconds")
