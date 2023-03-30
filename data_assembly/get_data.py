import numpy as np
import h5py as h5
import os,sys

import json


def rescale_kappa(f,g):
	kappa = f['kappa'][:]
	Omega_m = g['scalars/cosmology'][:]['omega_m'][0]
	kappa /= Omega_m
	return kappa

def kmode(n,N,L):
	if(n > N/2):
		a = (n-N)
	else: 
		a = n
	return 2.*np.pi / L * a

def get_shear(kappa, N0, N1, L0, L1, Ncat=4):
	shearR = np.zeros((Ncat,N0,N1))  
	shearI = np.zeros((Ncat,N0,N1))
	    
	for tomobin in range(0, Ncat, 1):
		array_out = np.fft.fft2(kappa[:,:,tomobin])

		array_in_complex = np.zeros((N0,N1),dtype=np.complex_)

		for n0 in range(0,N0,1):
			for n1 in range(0,N1,1):
				lx = kmode(n0,N0,L0)
				ly = kmode(n1,N1,L1)
				lnorm = lx*lx + ly*ly
				wave_vect = lx + 1j * ly
				array_in_complex[n0,n1] = wave_vect * wave_vect * array_out[n0,n1] / (lnorm+1e-20)
				

		shearR[tomobin,:,:] = np.real(np.fft.ifft2(array_in_complex))
		shearI[tomobin,:,:] = np.imag(np.fft.ifft2(array_in_complex))		
	return shearR + 1j * shearI


def get_data(f):
	dataR = np.zeros((4, 64, 64))
	dataI = np.zeros((4, 64, 64))

	bin0 = f['tomo0'][:]
	bin1 = f['tomo1'][:]
	bin2 = f['tomo2'][:]
	bin3 = f['tomo3'][:]

	data = np.stack([bin0,bin1,bin2,bin3])

	dataR = np.real(data)
	dataI = np.imag(data)

	return dataR, dataI

def load_fiducial(seed_index, path):
    
    data = []
    
    for i,seed in enumerate(seed_index):
        _dat = np.ones((8, 64, 64))

        f = h5.File(path + 'sim_fields_%d.h5'%(seed), 'r')

        print('loading file', path + 'sim_fields_%d.h5'%(seed))

        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI
        
        data.append(_dat)
        
    return np.array(data)




def load_derivatives(seed_index,
                     omegam_path,
                     sigma8_path,
                     Om_fid=0.3175,
                     s8_fid=0.800,
                     d_Om=0.05,
                     d_s8=0.015):
    
    Om_plus = Om_fid + d_Om
    Om_minus = Om_fid - d_Om

    s8_plus = s8_fid + d_s8
    s8_minus = s8_fid - d_s8
    
    data = []

    for s,seed in enumerate(seed_index):

        print('loading seed %d'%(s))
        # do each seed
        
        # '/data80/nporqueres/borg_sims/omegaM/seed_matching_06/sim_fields_0.55_%d.h5'%(seed)
        # first do OmegaM-
        _dat = np.ones((8, 64, 64))
        f = h5.File(omegam_path + 'sim_fields_%.04f_%d.h5'%(Om_minus, seed), 'r')
        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI
        
        data.append(_dat)

        # then sigma8-
        _dat = np.ones((8, 64, 64))
        f = h5.File(sigma8_path + 'sim_fields_%.03f_%d.h5'%(s8_minus, seed), 'r')

        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI

        data.append(_dat)#(np.concatenate((dataR, dataI), axis=0))) # [:4, :, :] is real, [4:, :, :] is im


        # then Omega+ 0.65
        _dat = np.ones((8, 64, 64))
        f = h5.File(omegam_path + 'sim_fields_%.04f_%d.h5'%(Om_plus, seed), 'r')
        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI

        data.append(_dat)

        # then sigma8 +
        _dat = np.ones((8, 64, 64))
        f = h5.File(sigma8_path + 'sim_fields_%.03f_%d.h5'%(s8_plus, seed), 'r')
        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI

        data.append(_dat)

                

    return np.array(data)





def load_h0(seed_index):
    # iterate over folders [Om_m, s8_m, Om_p, s8_p, ...]
    
    data = []

    for s,seed in enumerate(seed_index):

        print('loading seed %d'%(s))
        # do each seed
        
        # '/data80/nporqueres/borg_sims/omegaM/seed_matching_06/sim_fields_0.55_%d.h5'%(seed)
        # first do h0-
        _dat = np.ones((8, 64, 64))
        f = h5.File('/data80/nporqueres/borg_sims_fixed/h0/sim_fields_0.6511_%d.h5'%(seed), 'r')
        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI
        
        data.append(_dat)


        # then h0+
        _dat = np.ones((8, 64, 64))
        f = h5.File('/data80/nporqueres/borg_sims_fixed/h0/sim_fields_0.6911_%d.h5'%(seed), 'r')
        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI

        data.append(_dat)

                

    return np.array(data)


def load_delfi_data(seed_index, sim_fnames=None, 
                            sim_fieldfnames=None,
                            path="/data80/nporqueres/borg_sims/uniform_prior_sims/"):

    #data = np.zeros((num_sims,) + input_shape)
    data = []
    params = []

    if sim_fnames is not None:
        seed_index = sim_fnames
    
    for i,seed in enumerate(seed_index):
        _dat = np.ones((8, 64, 64))

        if sim_fnames is None:
            name = path + 'sim_%d.h5'%(seed)
            fieldname = path + 'sim_fields_%d.h5'%(seed)
            
        else:
            name = path + seed
            fieldname = path + sim_fieldfnames[i]

        print(name, fieldname)

        f = h5.File(fieldname, 'r')
        g = h5.File(name, 'r') # for cosmo params
        
        dataR, dataI = get_data(f)

        _dat[::2, :, :] = dataR
        _dat[1::2, :, :] = dataI
        
        data.append(_dat)

        # extract the theta from the prior draw
        Om = np.array(g['scalars/cosmology/'])[0][2]
        s8 = np.array(g['scalars/cosmology/'])[0][9]

        _params = np.array([Om, s8])

        params.append(_params)
        
    return np.array(data), np.array(params)


############################################################################

#if __name__ == 'main':

### ------------- READ IN CONFIGS -------------

config_path = sys.argv[1]
with open(config_path) as f:
        configs = json.load(f)


outdir = configs["datadir"]

fiducial_path = configs["borg_data_configs"]["fiducial_path"]
omegam_path = configs["borg_data_configs"]["omegaM_path"]
sigma8_path = configs["borg_data_configs"]["sigma8_path"]
prior_path = configs["borg_data_configs"]["prior_path"]

prior_save_dir = configs["priordir"]

omegam_stepsize = configs["borg_data_configs"]["omegaM_stepsize"]
sigma8_stepsize = configs["borg_data_configs"]["sigma8_stepsize"]


params = ['omegaM', 'sigma8']

θ_fid = np.array([0.3175, 0.800])
#δθ = 2*np.array([0.05, 0.015])

Om_fid = 0.3175
s8_fid = 0.800

dataset = sys.argv[2]

num_prior_sims = 10000



### ------------- PROCESS DATA -------------
print("BEGINNING BORG DATA PROCESSING")
print("SAVING TRAINING DATA TO: ", outdir)
print("SAVING PRIOR DATA TO: ", prior_save_dir)


if dataset == "fiducial":

    print("doing fiducial set")
    fid = load_fiducial(np.arange(1000), path=fiducial_path)
    val_fid = load_fiducial(np.arange(1000, 2000), path=fiducial_path)


    np.save(outdir + 'noisefree_fid', fid)
    np.save(outdir + 'noisefree_val_fid', val_fid)


elif dataset == "derivatives":

    print("doing derivatives")
    dervs = load_derivatives(np.arange(0, 250), omegam_path, sigma8_path, d_Om=omegam_stepsize)
    val_dervs = load_derivatives(np.arange(250, 500), omegam_path, sigma8_path, d_Om=omegam_stepsize)


    np.save(outdir + 'noisefree_derv_smallstep', dervs)
    np.save(outdir + 'noisefree_val_derv_smallstep', val_dervs)

elif dataset == "h0":

    print("getting h0 derivatives")

    outdir = '/data80/makinen/borg_sims_fixed/'

    h0_dervs = load_h0(np.arange(500))
    np.save(outdir + 'noisefree_h0_dervs', h0_dervs)

### ELSE DO PRIOR
else:

    print('doing prior from ', prior_path)


    dir_list = os.listdir(prior_path)

    simnames = []
    sim_fieldnames = []

    for s in dir_list:
        if "sim_" in s:
            if "fields" in s:
                sim_fieldnames.append(s)

            else:
                simnames.append(s)

    seed_index = []
    for s in simnames:
        seed_index.append(s[4:-3])

    seed_index = np.array(seed_index).astype(int)[:num_prior_sims]

    dat, params = load_delfi_data(seed_index=seed_index, 
                            sim_fnames=None, 
                            sim_fieldfnames=None,
                            path=prior_path)

    np.save(prior_save_dir + "noisefree_prior_sims", dat)
    np.save(prior_save_dir + "noisefree_prior_params", params)

    # first_idx = np.arange(1000)
    # next_idx = np.arange(2000, 5000)

    # full_idx = np.concatenate([first_idx, next_idx])

    # dat, params = load_delfi_data(seed_index=full_idx)

    # np.save(prior_save_dir + "noisefree_prior_sims", dat)
    # np.save(prior_save_dir + "noisefree_prior_params", params)